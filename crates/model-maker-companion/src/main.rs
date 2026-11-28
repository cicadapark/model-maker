//! model-maker-companion: a small window beside the studio page (docs/companion.md). It checks and sets this PC up,
//! makes one object locally and hands over its folder, and opens the studio; every step is the launcher's own.
//! It is drawn on the processor, so the card stays free for the make.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod app;
mod card;
mod children;
mod fonts;
mod make;
mod painter;
mod progress;
mod this_pc;

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::{Duration, Instant};

use egui_winit::accesskit_winit;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use painter::{Canvas, Painter};

/// The window's size in points (plan section 1).
const WINDOW_SIZE: (f64, f64) = (560.0, 700.0);
/// The smallest the window may be made: every row's words fit at this width (the pictures are drawn at it).
const WINDOW_MIN: (f64, f64) = (520.0, 560.0);
/// Text sizes in points: 16 and up (plan section 6).
const BODY_TEXT: f32 = 16.0;
/// While egui animates (a hover, an opening fold) frames come this often; otherwise only on a change.
const ANIMATION_FRAME: Duration = Duration::from_millis(33);

enum Wake {
    AccessKit(accesskit_winit::Event),
}

impl From<accesskit_winit::Event> for Wake {
    fn from(event: accesskit_winit::Event) -> Wake {
        Wake::AccessKit(event)
    }
}

struct Shown {
    window: Rc<Window>,
    surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    state: egui_winit::State,
}

struct Shell {
    app: app::App,
    context: egui::Context,
    painter: Painter,
    shown: Option<Shown>,
    proxy: EventLoopProxy<Wake>,
    next_frame: Option<Instant>,
    next_poll: Instant,
}

fn styled(context: &egui::Context) {
    context.all_styles_mut(|style| {
        use egui::{FontFamily, FontId, TextStyle};
        style.text_styles = [
            (
                TextStyle::Small,
                FontId::new(BODY_TEXT - 2.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Body,
                FontId::new(BODY_TEXT, FontFamily::Proportional),
            ),
            (
                TextStyle::Button,
                FontId::new(BODY_TEXT, FontFamily::Proportional),
            ),
            (
                TextStyle::Heading,
                FontId::new(BODY_TEXT + 4.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(BODY_TEXT - 2.0, FontFamily::Monospace),
            ),
        ]
        .into();
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        // A box or button's edge is visible in both themes (light's own is white on near-white).
        let edge = style.visuals.widgets.noninteractive.bg_stroke.color;
        style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, edge);
    });
}

impl Shell {
    fn redraw(&mut self) {
        let Some(shown) = &mut self.shown else { return };
        let input = shown.state.take_egui_input(&shown.window);
        let mut output = self.context.run_ui(input, |ui| self.app.ui(ui));
        let mut textures = std::mem::take(&mut output.textures_delta);
        shown
            .state
            .handle_platform_output(&shown.window, output.platform_output);
        self.painter.update_textures(&textures);
        let primitives = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
        let size = shown.window.inner_size();
        if let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        {
            if shown.surface.resize(width, height).is_ok() {
                let background = self
                    .context
                    .style_of(self.context.theme())
                    .visuals
                    .panel_fill;
                // Without the memory for this picture (the computer is out of it), this frame is skipped: the window
                // stays, and draws again once memory is back.
                if let Some(mut canvas) =
                    Canvas::try_new(size.width as usize, size.height as usize, background)
                {
                    self.painter
                        .paint(&mut canvas, &primitives, output.pixels_per_point);
                    if let Ok(mut buffer) = shown.surface.buffer_mut() {
                        canvas.write_0rgb(&mut buffer);
                        let _ = buffer.present();
                    }
                } else {
                    shown.window.request_redraw();
                }
            }
        }
        self.painter.free_textures(&textures);
        // Applied: egui checks that no delta is dropped unapplied.
        textures.clear();
        let delay = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);
        self.next_frame = (delay < Duration::from_secs(3600))
            .then(|| Instant::now() + delay.max(ANIMATION_FRAME));
    }
}

impl ApplicationHandler<Wake> for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.shown.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("model-maker companion")
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_SIZE.0, WINDOW_SIZE.1))
            .with_min_inner_size(winit::dpi::LogicalSize::new(WINDOW_MIN.0, WINDOW_MIN.1))
            // Shown once AccessKit is attached (its adapter must come before the window is visible).
            .with_visible(false);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Rc::new(window),
            Err(error) => {
                eprintln!("model-maker-companion: no window could be made ({error})");
                event_loop.exit();
                return;
            }
        };
        let surface = softbuffer::Context::new(window.clone())
            .and_then(|context| softbuffer::Surface::new(&context, window.clone()));
        let surface = match surface {
            Ok(surface) => surface,
            Err(error) => {
                eprintln!("model-maker-companion: the window cannot be drawn ({error})");
                event_loop.exit();
                return;
            }
        };
        let mut state = egui_winit::State::new(
            self.context.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        state.init_accesskit(event_loop, &window, self.proxy.clone());
        window.set_visible(true);
        window.request_redraw();
        self.shown = Some(Shown {
            window,
            surface,
            state,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(shown) = &mut self.shown else { return };
        let response = shown.state.on_window_event(&shown.window, &event);
        match event {
            WindowEvent::CloseRequested => {
                self.app.ask_to_quit();
                shown.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ if response.repaint => shown.window.request_redraw(),
            _ => {}
        }
        if self.app.quit {
            event_loop.exit();
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, event: Wake) {
        let Some(shown) = &mut self.shown else { return };
        let Wake::AccessKit(event) = event;
        match event.window_event {
            accesskit_winit::WindowEvent::InitialTreeRequested => self.context.enable_accesskit(),
            accesskit_winit::WindowEvent::ActionRequested(request) => {
                shown.state.on_accesskit_action_request(request)
            }
            accesskit_winit::WindowEvent::AccessibilityDeactivated => {}
        }
        shown.window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let mut draw = self.next_frame.is_some_and(|at| at <= now);
        if now >= self.next_poll {
            draw |= self.app.poll();
            self.next_poll = now + self.app.tick();
        }
        if draw {
            self.next_frame = None;
            if let Some(shown) = &self.shown {
                shown.window.request_redraw();
            }
        }
        let wake = self
            .next_frame
            .map_or(self.next_poll, |at| at.min(self.next_poll));
        event_loop.set_control_flow(ControlFlow::WaitUntil(wake));
    }
}

/// One companion at a time on this PC: a second one would start a second make on the same card.
fn single(work: &std::path::Path) -> Result<Option<std::fs::File>, String> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(work.join("companion.lock"))
        .map_err(|e| format!("the lock file could not be opened ({e})"))?;
    file.try_lock()
        .map_err(|_| "The companion is already open for this model-maker folder.".to_string())?;
    Ok(Some(file))
}

fn main() -> std::process::ExitCode {
    if let Err(why) = model_maker_launcher::run::own_children() {
        eprintln!("model-maker-companion: programs it starts may outlive it ({why})");
    }
    let Some((font, index)) = fonts::system_font() else {
        eprintln!("model-maker-companion: this system has no font it can read; nothing is shipped, so it stops");
        return std::process::ExitCode::from(2);
    };
    let this_program = std::env::current_exe()
        .ok()
        .map(|p| model_maker_launcher::paths::real(&p));
    let repository = model_maker_launcher::repository::of_this_program().map_err(|e| e.to_string());
    let launcher = this_program.as_deref().and_then(children::launcher_beside);
    // The window's own folder is the repository's (its tools folder, this user's only); without a repository the
    // window only says why, and writes nothing.
    let (work, lock) = match repository.as_deref().map(this_pc::own_folder) {
        Ok(Ok(work)) => {
            let lock = single(&work);
            if lock.is_ok() {
                this_pc::remove_token(&work.join(this_pc::TOKEN_FILE));
            }
            (work, lock)
        }
        Ok(Err(why)) => (std::env::temp_dir(), Err(why)),
        Err(_) => (std::env::temp_dir(), Ok(None)),
    };
    let mut app = app::App::new(repository, launcher, work);
    if let Err(why) = &lock {
        app.refuse(why);
    }
    let context = egui::Context::default();
    context.set_fonts(fonts::definitions(font, index));
    styled(&context);
    let event_loop = match EventLoop::<Wake>::with_user_event().build() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            eprintln!("model-maker-companion: no window system ({error})");
            return std::process::ExitCode::from(2);
        }
    };
    let mut shell = Shell {
        app,
        context,
        painter: Painter::default(),
        shown: None,
        proxy: event_loop.create_proxy(),
        next_frame: None,
        next_poll: Instant::now(),
    };
    let ran = event_loop.run_app(&mut shell);
    drop(lock);
    match ran {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("model-maker-companion: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Pictures of the window's states, drawn exactly as the window draws them but without a window (the same painter):
/// `COMPANION_PICTURES=<folder> cargo test -p model-maker-companion -- --ignored pictures --test-threads 1`.
#[cfg(all(test, unix))]
mod pictures {
    use super::*;
    use model_maker_launcher::check::{Row, Verdict};
    use model_maker_launcher::setup;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    fn ppm(canvas: &Canvas) -> Vec<u8> {
        let mut out = format!("P6\n{} {}\n255\n", canvas.width, canvas.height).into_bytes();
        for pixel in &canvas.pixels {
            out.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b()]);
        }
        out
    }

    fn frame(
        context: &egui::Context,
        app: &mut app::App,
        painter: &mut Painter,
        size: egui::Vec2,
        time: f64,
    ) -> Canvas {
        // The clock moves a second a frame, so a dialog's fade-in has ended by the picture.
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            time: Some(time),
            ..Default::default()
        };
        let mut output = context.run_ui(input, |ui| app.ui(ui));
        let mut textures = std::mem::take(&mut output.textures_delta);
        painter.update_textures(&textures);
        let primitives = context.tessellate(output.shapes, output.pixels_per_point);
        let background = context.style_of(context.theme()).visuals.panel_fill;
        let mut canvas = Canvas::new(size.x as usize, size.y as usize, background);
        painter.paint(&mut canvas, &primitives, output.pixels_per_point);
        painter.free_textures(&textures);
        textures.clear();
        canvas
    }

    /// A stand-in launcher: setup and makes that end each way the window shows.
    const STAND_IN: &str = r#"#!/bin/sh
if [ "$1" = "setup" ]; then
  echo '{"schema":1,"event":"began","step":"python"}' >> "$MODEL_MAKER_PROGRESS"
  echo "uv: installing the locked environment"
  echo '{"schema":1,"event":"began","step":"tools"}' >> "$MODEL_MAKER_PROGRESS"
  echo '{"schema":1,"event":"began","step":"models"}' >> "$MODEL_MAKER_PROGRESS"
  [ "$2" = "--yes" ] && exec sleep 30
  echo '{"schema":1,"event":"stopped","why":"FLUX.1 [schnell]: the hub needs a signed-in account for it and no token was given"}' >> "$MODEL_MAKER_PROGRESS"
  exit 1
fi
out="$4"; job="$out/chair-1"; ledger="$REPO/metrics/jobs.jsonl"; batch=$(basename "$out"); mkdir -p "$job"
line() { printf '{"event":"stage","stage":"%s","seconds":%s,"batch":"%s","folder":"%s"}\n' "$1" "$2" "$batch" "$job" >> "$ledger"; }
case "$2" in
  "a sword of war") printf '{"event":"not made","why":"a weapon of war is not made","batch":"%s","folder":null}\n' "$batch" >> "$ledger"; exit 0;;
esac
line pictures 95; line shape 140
echo "pictures: 4 tries, the second kept"
case "$2" in running|cancelled) exec sleep 30;; esac
line paint 300; line finish 280
if [ "$2" = "a cracked bowl" ]; then
  printf '{"checks":{"shape":[{"name":"stands","passed":true},{"name":"one piece","passed":false}],"paint":[{"name":"colours","passed":false},{"name":"no lettering","passed":true}]}}' > "$job/chair.provenance.json"
else
  printf '{"checks":{"shape":[{"name":"stands","passed":true},{"name":"one piece","passed":true}],"paint":[{"name":"colours","passed":true}]}}' > "$job/chair.provenance.json"
fi
printf '{"event":"job","batch":"%s","folder":"%s","slug":"chair","passed":true,"failed":[]}\n' "$batch" "$job" >> "$ledger"
"#;

    fn row(check: &'static str, found: &str, verdict: Verdict) -> Row {
        Row {
            check,
            found: found.into(),
            verdict,
        }
    }

    fn fresh_pc(missing: bool) -> this_pc::Checked {
        let ok = |w: &str| Verdict::Ok(w.into());
        let mut rows = vec![
            row("System", "Linux", ok("ok")),
            row("Graphics card", "NVIDIA GeForce RTX 3060, 12 GB", ok("ok")),
            row(
                "Memory",
                if missing { "8 GB" } else { "32 GB" },
                if missing {
                    Verdict::Missing("16 GB at least for a make".into())
                } else {
                    ok("ok")
                },
            ),
            row("Setup", "no record of a finished setup", ok("run `model-maker setup` (once more on a PC set up before: it carries on where it stopped)")),
            row(
                "This program",
                "unsigned build",
                ok("check the archive's sha256 against the published one"),
            ),
        ];
        rows.rotate_left(0);
        this_pc::Checked {
            missing: model_maker_launcher::check::missing_count(&rows),
            rows,
            environment_here: false,
            runtime_missing: false,
            setup_done: setup::Done::No,
        }
    }

    fn ready_pc() -> this_pc::Checked {
        let mut checked = fresh_pc(false);
        checked.environment_here = true;
        checked.setup_done = setup::Done::WithModels;
        checked
    }

    struct Scene {
        name: &'static str,
        dark: bool,
        width: f32,
    }

    #[test]
    #[ignore = "needs the system's font; writes pictures for a person to look at"]
    fn pictures_of_every_state() {
        let folder = std::env::var_os("COMPANION_PICTURES")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("companion-pictures"));
        let stand_in = folder.join("stand-in");
        let _ = std::fs::remove_dir_all(&stand_in);
        std::fs::create_dir_all(stand_in.join("repo/metrics")).unwrap();
        let launcher = stand_in.join("model-maker");
        let script = STAND_IN.replace("$REPO", &stand_in.join("repo").display().to_string());
        std::fs::write(&launcher, script).unwrap();
        std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (font, index) = fonts::system_font().expect("a system font");
        let repository =
            std::fs::canonicalize(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).unwrap();
        let scenes = [
            ("fresh-this-pc", false),
            ("fresh-this-pc", true),
            ("missing-row", false),
            ("licence-open", false),
            ("studio", false),
            ("make-not-ready", true),
            ("make-form-more", false),
            ("card-open", false),
            ("setup-running", false),
            ("setup-running", true),
            ("setup-stopped", false),
            ("question-stop-setup", false),
            ("question-cancel-make", true),
            ("question-quit", false),
            ("make-running", false),
            ("make-running", true),
            ("make-done", false),
            ("make-failed-checks", false),
            ("make-not-made", false),
            ("make-cancelled", false),
            ("second-window", false),
            ("details-open", false),
        ];
        for (name, dark) in scenes {
            for width in [WINDOW_MIN.0 as f32, WINDOW_SIZE.0 as f32] {
                if width == WINDOW_SIZE.0 as f32
                    && name != "fresh-this-pc"
                    && name != "make-running"
                {
                    continue;
                }
                let scene = Scene { name, dark, width };
                draw(
                    &scene,
                    &folder,
                    &stand_in,
                    &launcher,
                    &repository,
                    &font,
                    index,
                );
            }
        }
    }

    fn draw(
        scene: &Scene,
        folder: &std::path::Path,
        stand_in: &std::path::Path,
        launcher: &std::path::Path,
        repository: &std::path::Path,
        font: &[u8],
        index: u32,
    ) {
        let context = egui::Context::default();
        context.set_fonts(fonts::definitions(font.to_vec(), index));
        styled(&context);
        context.set_theme(if scene.dark {
            egui::Theme::Dark
        } else {
            egui::Theme::Light
        });
        let mut app = app::App::new(
            Ok(repository.to_path_buf()),
            Some(launcher.to_path_buf()),
            stand_in.join("work"),
        );
        let since = Instant::now();
        while !app.checked_yet() && since.elapsed() < Duration::from_secs(60) {
            app.poll();
            std::thread::sleep(Duration::from_millis(100));
        }
        app.checked = Some(fresh_pc(scene.name == "missing-row"));
        app.tab = app::Tab::ThisPc;
        app.open_folds = matches!(
            scene.name,
            "licence-open" | "card-open" | "make-form-more" | "details-open"
        );
        *app.card.latest.lock().unwrap() =
            card::Card::Read(card::parse("NVIDIA GeForce RTX 3060, 97, 71, 11012, 12288").unwrap());
        let mut prompt = "an old looking chair";
        match scene.name {
            "studio" => app.tab = app::Tab::Studio,
            "make-not-ready" => app.tab = app::Tab::Make,
            "make-form-more" => {
                app.checked = Some(ready_pc());
                app.tab = app::Tab::Make;
                app.form.prompt = "an old looking chair".into();
                app.form.size = "0.9".into();
            }
            "second-window" => {
                app.refuse("The companion is already open for this model-maker folder.")
            }
            "setup-running" | "setup-stopped" | "question-stop-setup" | "details-open" => {
                let choices = this_pc::Choices {
                    licences_accepted: scene.name != "setup-stopped",
                    ..Default::default()
                };
                let setup = this_pc::Setup::start(
                    launcher,
                    repository,
                    &stand_in.join("work"),
                    &choices,
                    57_300_000_000,
                    false,
                )
                .unwrap();
                app.setup = Some(setup);
                if scene.name == "question-stop-setup" {
                    app.question = Some(app::Question::StopSetup);
                }
            }
            name if name.starts_with("make-") || name == "question-cancel-make" => {
                prompt = match name {
                    "make-running" | "question-cancel-make" => "running",
                    "make-cancelled" => "cancelled",
                    "make-failed-checks" => "a cracked bowl",
                    "make-not-made" => "a sword of war",
                    _ => "an old looking chair",
                };
                app.checked = Some(ready_pc());
                app.tab = app::Tab::Make;
                app.form = make::Form {
                    prompt: prompt.into(),
                    out: stand_in.join("made").display().to_string(),
                    ..make::Form::default()
                };
                let making =
                    make::Making::start(launcher, &stand_in.join("repo"), &app.form).unwrap();
                app.making = Some(making);
                if name == "question-cancel-make" {
                    app.question = Some(app::Question::CancelMake);
                }
            }
            "question-quit" => {
                app.question = Some(app::Question::Quit);
            }
            _ => {}
        }
        if prompt != "an old looking chair" {
            app.form.prompt = prompt.into();
        }
        std::thread::sleep(Duration::from_millis(1200));
        if scene.name == "make-cancelled" {
            if let Some(making) = &mut app.making {
                making.poll();
                making.running.stop();
            }
            std::thread::sleep(Duration::from_millis(1500));
        }
        for _ in 0..3 {
            if let Some(setup) = &mut app.setup {
                setup.poll();
            }
            if let Some(making) = &mut app.making {
                making.poll();
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        // The licences' list is drawn whole, its window as tall as it needs.
        let height = if scene.name == "licence-open" {
            2200.0
        } else {
            WINDOW_SIZE.1 as f32
        };
        let size = egui::vec2(scene.width, height);
        let mut painter = Painter::default();
        let mut canvas = frame(&context, &mut app, &mut painter, size, 0.0);
        for second in 1..5 {
            canvas = frame(&context, &mut app, &mut painter, size, f64::from(second));
        }
        let theme = if scene.dark { "dark" } else { "light" };
        let file = folder.join(format!("{}-{theme}-{}.ppm", scene.name, scene.width as u32));
        std::fs::write(&file, ppm(&canvas)).unwrap();
        println!("{}", file.display());
    }
}

/// Every control is named for a screen reader (AccessKit), in each tab (plan section 6).
#[cfg(test)]
mod named_for_screen_readers {
    use super::*;

    /// A node's text as a screen reader reads it: its label, else its value, else its children's text (egui's label
    /// keeps its words in TextRun children).
    fn text_of(
        nodes: &std::collections::HashMap<egui::accesskit::NodeId, &egui::accesskit::Node>,
        id: egui::accesskit::NodeId,
    ) -> String {
        let Some(node) = nodes.get(&id) else {
            return String::new();
        };
        if let Some(text) = node.label().or(node.value()).filter(|t| !t.is_empty()) {
            return text.to_string();
        }
        node.children()
            .iter()
            .map(|child| text_of(nodes, *child))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Each node's role and name: its own label, else the text of the nodes it is labelled by (as a screen reader
    /// reads it).
    fn names(tab: app::Tab) -> Option<Vec<(egui::Role, String)>> {
        let (font, index) = fonts::system_font()?;
        let context = egui::Context::default();
        context.set_fonts(fonts::definitions(font, index));
        styled(&context);
        context.enable_accesskit();
        let work = std::env::temp_dir().join(format!("companion-a11y-{}", std::process::id()));
        let repository =
            std::fs::canonicalize(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).unwrap();
        let mut app = app::App::new(Ok(repository), None, work);
        app.tab = tab;
        let mut found = Vec::new();
        for _ in 0..3 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(520.0, 680.0),
                )),
                ..Default::default()
            };
            let mut output = context.run_ui(input, |ui| app.ui(ui));
            output.textures_delta.clear();
            if let Some(update) = output.platform_output.accesskit_update {
                let nodes: std::collections::HashMap<_, _> =
                    update.nodes.iter().map(|(id, node)| (*id, node)).collect();
                found = update
                    .nodes
                    .iter()
                    .map(|(_, node)| {
                        let own = node.label().unwrap_or("").to_string();
                        let name = if own.is_empty() {
                            node.labelled_by()
                                .iter()
                                .map(|other| text_of(&nodes, *other))
                                .collect::<Vec<_>>()
                                .join(" ")
                        } else {
                            own
                        };
                        (node.role(), name)
                    })
                    .collect();
            }
        }
        Some(found)
    }

    #[test]
    fn buttons_tabs_and_boxes_have_names() {
        let Some(make) = names(app::Tab::Make) else {
            // CI sets COMPANION_REQUIRE_FONT: there a missing font fails instead of skipping.
            assert!(
                std::env::var_os("COMPANION_REQUIRE_FONT").is_none(),
                "no system font on this runner"
            );
            eprintln!("no system font here: skipped");
            return;
        };
        let named =
            |role: egui::Role, label: &str| make.iter().any(|(r, l)| *r == role && l == label);
        assert!(named(egui::Role::Button, "Open studio"), "{make:?}");
        assert!(named(egui::Role::Button, "Make"), "{make:?}");
        let unnamed = make
            .iter()
            .find(|(r, l)| *r == egui::Role::Button && l.is_empty());
        assert!(unnamed.is_none(), "a button without a name: {unnamed:?}");
        // The Windows test of 28 Nov 2026: the make form's three boxes had no name.
        for label in [
            "What to make (one line):",
            "Largest side in metres (empty: usual):",
            "The folder it goes in:",
        ] {
            assert!(named(egui::Role::TextInput, label), "{label}: {make:?}");
        }
        let boxes = make.iter().filter(|(r, _)| *r == egui::Role::TextInput);
        assert!(
            boxes.clone().all(|(_, l)| !l.is_empty()),
            "a box without a name: {:?}",
            boxes.collect::<Vec<_>>()
        );
        let this_pc = names(app::Tab::ThisPc).unwrap();
        assert!(this_pc.iter().any(|(_, l)| l == "This PC"));
    }
}
