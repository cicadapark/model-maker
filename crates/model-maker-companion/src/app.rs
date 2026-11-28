//! The window's one screen: the status line on top, the tabs This PC, Studio and Make, and "Open studio" at the bottom
//! (plan section 1). Every button is a call the command line also makes; the window adds no ability.
//!
//! Loading bars, no jumps: each part keeps its rows from the first frame (a row's words change, its place does not);
//! questions come as a dialog over the window; what ends a setup or a make is written last, below everything else.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use egui::{Button, CollapsingHeader, ProgressBar, RichText, ScrollArea, TextEdit, Ui};
use model_maker_launcher::check::{self, Verdict};

use crate::card;
use crate::make::{self, Choice, Form, Making, STAGES};
use crate::progress;
use crate::this_pc::{self, Checked, Choices, Licence, Outcome, Setup, Step};

/// The one address the program opens (never one sent to it).
pub const STUDIO_ADDRESS: &str = "https://cicadapark.com/model-maker-studio/";
pub const STUDIO_NEXT_VERSION: &str = "Connecting to the studio comes in the next version.";
const ROW_HEIGHT: f32 = 22.0;
const BAR_WIDTH: f32 = 120.0;
/// The names column of the step and stage rows.
const NAME_WIDTH: f32 = 100.0;
/// The check's row that says the program is unsigned (shown on the first screen, not only in the fold).
const PROGRAM_ROW: &str = "This program";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    ThisPc,
    Studio,
    Make,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Question {
    StopSetup,
    CancelMake,
    Quit,
}

pub struct App {
    pub(crate) repository: Result<PathBuf, String>,
    pub(crate) launcher: Option<PathBuf>,
    pub(crate) work: PathBuf,
    pub tab: Tab,
    checking: Option<Receiver<Checked>>,
    pub(crate) checked: Option<Checked>,
    licences: Result<Vec<Licence>, String>,
    pub(crate) accepted: bool,
    install_runtime: bool,
    pub(crate) token: String,
    pub(crate) setup: Option<Setup>,
    pub(crate) form: Form,
    pub(crate) making: Option<Making>,
    pub(crate) problem: Option<String>,
    pub(crate) question: Option<Question>,
    pub quit: bool,
    pub(crate) card: card::Watch,
    /// A second companion: it shows only why, and starts nothing.
    refused: Option<String>,
    /// For the review pictures only: every fold drawn open.
    pub(crate) open_folds: bool,
}

fn step_title(id: &str) -> &'static str {
    match id {
        "runtime" => "C++ runtime",
        "python" => "Python",
        "tools" => "Tools",
        "models" => "Models",
        "self-test" => "Self-test",
        _ => "Step",
    }
}

fn stage_title(stage: &str) -> &'static str {
    match stage {
        "pictures" => "Pictures",
        "shape" => "Shape",
        "paint" => "Paint",
        _ => "Finish",
    }
}

/// A row of fixed height, so nothing shifts when its words change.
/// Its words wrap within the window's width rather than widen the page (a row that never wraps would set the
/// width every other label wraps at).
fn fixed_row(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    ui.allocate_ui(egui::vec2(ui.available_width(), ROW_HEIGHT), |ui| {
        ui.set_min_height(ROW_HEIGHT);
        ui.horizontal_wrapped(add);
    });
}

/// The names column: left-aligned, one width for every row.
fn name_cell(ui: &mut Ui, name: &str) {
    ui.allocate_ui_with_layout(
        egui::vec2(NAME_WIDTH, ROW_HEIGHT),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.set_min_width(NAME_WIDTH);
            ui.label(name);
        },
    );
}

fn bar(ui: &mut Ui, fraction: f32) {
    ui.add(ProgressBar::new(fraction).desired_width(BAR_WIDTH));
}

/// The check's verdict in words: the launcher's own words, with "ok" said once.
pub fn verdict_words(row: &check::Row) -> String {
    match &row.verdict {
        Verdict::Ok(words) if words.starts_with("ok") => {
            format!("{}: {} ({words})", row.check, row.found)
        }
        Verdict::Ok(words) => format!("{}: {} (ok: {words})", row.check, row.found),
        Verdict::Missing(needed) => format!("{}: {} (MISSING: {needed})", row.check, row.found),
    }
}

/// Why "Set up" cannot start yet; None when it can. Nothing is enabled before the first check has come back.
pub fn setup_gate(
    checked: Option<&Checked>,
    launcher_here: bool,
    make_running: bool,
    accepted: bool,
) -> Option<&'static str> {
    let Some(checked) = checked else {
        return Some("Checking this PC first…");
    };
    if this_pc::blocking(&checked.rows) > 0 {
        return Some("Something is missing: see the list above.");
    }
    if !launcher_here {
        return Some("model-maker is not beside this program.");
    }
    if make_running {
        return Some("A make is running.");
    }
    if !accepted {
        return Some("Accept the licences first.");
    }
    None
}

/// Only an https address is handed to the system's browser (the registry's licence pages, the studio).
pub fn openable(address: &str) -> bool {
    address.starts_with("https://") && !address.contains(char::is_whitespace)
}

/// A model's name for its fold's header, short enough for one line: up to its first " (" or ", "
/// ("Qwen 3.5 9B (quantised by bartowski, with its vision part)" is "Qwen 3.5 9B"); the fold shows the whole name.
fn short_name(name: &str) -> &str {
    let end = [name.find(" ("), name.find(", ")]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(name.len());
    &name[..end]
}

impl App {
    pub fn new(
        repository: Result<PathBuf, String>,
        launcher: Option<PathBuf>,
        work: PathBuf,
    ) -> App {
        let licences = match &repository {
            Ok(r) => this_pc::read_licences(r),
            Err(why) => Err(why.clone()),
        };
        let mut app = App {
            repository,
            launcher,
            work,
            tab: Tab::ThisPc,
            checking: None,
            checked: None,
            licences,
            accepted: false,
            install_runtime: false,
            token: String::new(),
            setup: None,
            form: Form::default(),
            making: None,
            problem: None,
            question: None,
            quit: false,
            card: card::Watch::start(),
            refused: None,
            open_folds: false,
        };
        app.check_again();
        app
    }

    fn check_again(&mut self) {
        let Ok(repository) = self.repository.clone() else {
            return;
        };
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = send.send(this_pc::checked(&repository));
        });
        self.checking = Some(receive);
    }

    fn models_wanted(&self) -> u64 {
        self.licences.as_deref().map_or(0, this_pc::bytes_wanted)
    }

    /// The first check has come back.
    #[cfg(all(test, unix))]
    pub fn checked_yet(&self) -> bool {
        self.checked.is_some()
    }

    pub fn ready(&self) -> bool {
        self.checked.as_ref().is_some_and(this_pc::ready)
    }

    fn setup_running(&self) -> bool {
        self.setup.as_ref().is_some_and(|s| s.outcome.is_none())
    }

    fn make_running(&self) -> bool {
        self.making.as_ref().is_some_and(|m| m.ended.is_none())
    }

    /// Something runs that closing the window would stop.
    pub fn busy(&self) -> bool {
        self.setup_running() || self.make_running()
    }

    /// Reads the children's news; true when the screen should be drawn again.
    pub fn poll(&mut self) -> bool {
        if self.refused.is_some() {
            return false;
        }
        let mut changed = false;
        if let Some(receive) = &self.checking {
            if let Ok(checked) = receive.try_recv() {
                self.checked = Some(checked);
                self.checking = None;
                if self.ready() && self.setup.is_none() && self.tab == Tab::ThisPc {
                    self.tab = Tab::Make;
                }
                changed = true;
            }
        }
        if let Some(setup) = &mut self.setup {
            if setup.outcome.is_none() {
                changed = true;
                if !setup.poll() {
                    self.check_again();
                }
            }
        }
        if let Some(making) = &mut self.making {
            if making.ended.is_none() {
                making.poll();
                changed = true;
            }
        }
        changed
    }

    pub fn ask_to_quit(&mut self) {
        if self.busy() {
            self.question = Some(Question::Quit);
        } else {
            self.quit = true;
        }
    }

    pub fn refuse(&mut self, why: &str) {
        self.refused = Some(why.to_string());
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        if let Some(why) = &self.refused {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.label(RichText::new(why).strong());
                ui.label("Use the window that is already open, or close it first.");
            });
            return;
        }
        egui::Panel::top("status").show(ui, |ui| self.status(ui));
        egui::Panel::bottom("open-studio").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Colours:");
                let mut preference = ui.ctx().options(|o| o.theme_preference);
                for (choice, words) in [
                    (egui::ThemePreference::System, "as the system"),
                    (egui::ThemePreference::Light, "light"),
                    (egui::ThemePreference::Dark, "dark"),
                ] {
                    ui.selectable_value(&mut preference, choice, words);
                }
                ui.ctx().set_theme(preference);
            });
            let open = Button::new(RichText::new("Open studio").size(20.0))
                .min_size(egui::vec2(ui.available_width(), 44.0));
            if ui.add(open).clicked() {
                self.open_address(STUDIO_ADDRESS);
            }
            ui.add_space(6.0);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (tab, words) in [
                    (Tab::ThisPc, "This PC"),
                    (Tab::Studio, "Studio"),
                    (Tab::Make, "Make"),
                ] {
                    ui.selectable_value(&mut self.tab, tab, RichText::new(words).size(18.0));
                }
            });
            // The problem line keeps its row when there is none.
            fixed_row(ui, |ui| {
                ui.label(RichText::new(self.problem.as_deref().unwrap_or("")).strong());
            });
            ui.separator();
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match self.tab {
                    Tab::ThisPc => self.this_pc(ui),
                    Tab::Studio => {
                        ui.label(STUDIO_NEXT_VERSION);
                        ui.label("Until then, the studio page opens with the button below.");
                    }
                    Tab::Make => self.make_tab(ui),
                });
        });
        if let Some(question) = self.question {
            self.asked(ui.ctx(), question);
        }
    }

    fn open_address(&mut self, address: &str) {
        if !openable(address) {
            self.problem = Some("Only https addresses are opened.".to_string());
            return;
        }
        self.problem = open::that_detached(address)
            .err()
            .map(|e| format!("The page could not be opened ({e})."));
    }

    /// A question as a dialog over the window: nothing under it moves.
    fn asked(&mut self, ctx: &egui::Context, question: Question) {
        let (words, yes, no) = question_words(question);
        let mut answer = None;
        egui::Modal::new(egui::Id::unique("question")).show(ctx, |ui| {
            ui.set_max_width(360.0);
            ui.label(RichText::new(words).strong());
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button(yes).clicked() {
                    answer = Some(true);
                }
                if ui.button(no).clicked() {
                    answer = Some(false);
                }
            });
        });
        match answer {
            Some(true) => {
                self.question = None;
                match question {
                    Question::StopSetup => {
                        if let Some(setup) = &mut self.setup {
                            setup.running.stop();
                        }
                    }
                    Question::CancelMake => {
                        if let Some(making) = &mut self.making {
                            making.running.stop();
                        }
                    }
                    Question::Quit => self.quit = true,
                }
            }
            Some(false) => self.question = None,
            None => {}
        }
    }

    fn status(&mut self, ui: &mut Ui) {
        let job = if let Some(setup) = self.setup.as_ref().filter(|s| s.outcome.is_none()) {
            let step = setup
                .steps
                .iter()
                .find(|(_, s)| matches!(s, Step::Running(_)))
                .map_or("starting", |(id, _)| step_title(id));
            format!("Setting up this PC: {step}")
        } else if let Some(making) = self.making.as_ref().filter(|m| m.ended.is_none()) {
            let stage = making
                .job
                .current()
                .map_or("finishing", |i| stage_title(STAGES[i]));
            format!("Making your object: {stage}")
        } else {
            "Waiting: nothing is running".to_string()
        };
        fixed_row(ui, |ui| {
            ui.label(RichText::new(job).strong());
        });
        fixed_row(ui, |ui| {
            ui.label("Studio: not connected (no studio link in this version)");
        });
        CollapsingHeader::new("The card")
            .default_open(self.open_folds)
            .show(ui, |ui| {
                for (what, value) in card::rows(&self.card.now()) {
                    fixed_row(ui, |ui| {
                        ui.label(format!("{what}: {value}"));
                    });
                }
            });
    }

    // --- This PC ---

    fn this_pc(&mut self, ui: &mut Ui) {
        if let Err(why) = &self.repository {
            ui.label(why.as_str());
            return;
        }
        let state = match &self.checked {
            None => "Checking this PC (nothing is changed)…".to_string(),
            Some(_) if self.ready() => "This PC is ready.".to_string(),
            Some(c) if c.missing > 0 => {
                format!("{} thing(s) missing: see the list below.", c.missing)
            }
            Some(c) if c.setup_done == model_maker_launcher::setup::Done::WithoutModels => {
                "Set up without the models: run setup to fetch them.".to_string()
            }
            Some(_) => "No record of a finished setup: run setup (once more if this PC was set up before; it \
                        carries on where it stopped)."
                .to_string(),
        };
        fixed_row(ui, |ui| {
            ui.label(RichText::new(state).strong());
        });
        let unsigned = self
            .checked
            .as_ref()
            .and_then(|c| c.rows.iter().find(|r| r.check == PROGRAM_ROW))
            .map_or(String::new(), verdict_words);
        // Its words never change once read, so it wraps freely (a row that never wraps would widen the whole page).
        ui.label(RichText::new(unsigned).small());
        // While setup runs, its rows come first.
        if self.setup.is_some() {
            self.setup_rows(ui);
            ui.separator();
        }
        match &self.checked {
            None => fixed_row(ui, |ui| {
                ui.label("What was checked: checking…");
            }),
            Some(checked) => {
                CollapsingHeader::new("What was checked")
                    .id_salt("what-was-checked")
                    .default_open(checked.missing > 0 || self.open_folds)
                    .show(ui, |ui| {
                        for row in &checked.rows {
                            let words = RichText::new(verdict_words(row));
                            ui.label(if matches!(row.verdict, Verdict::Missing(_)) {
                                words.strong()
                            } else {
                                words
                            });
                        }
                    });
            }
        }
        ui.separator();
        if self.setup.is_none() && !self.ready() {
            self.before_setup(ui);
        }
    }

    fn before_setup(&mut self, ui: &mut Ui) {
        ui.heading("Set up");
        let licences = self.licences.clone();
        let licences = match &licences {
            Err(why) => {
                ui.label(format!("The models' list could not be read: {why}"));
                return;
            }
            Ok(licences) => licences,
        };
        ui.label(format!(
            "Setup downloads the models listed below ({}). Read each one's licence first.",
            progress::gigabytes(this_pc::bytes_wanted(licences))
        ));
        // The choices and the button first, so they are on the first screen; the list follows.
        ui.checkbox(
            &mut self.accepted,
            "I have read the licences below and accept them",
        );
        if licences.iter().any(|l| l.gated) {
            let words = ui.label(
                "FLUX.1 [schnell] is given only to a signed-in Hugging Face account that accepted its terms on its \
                 page. That account's access token (read only):",
            );
            // Each box is named by the words above it, for a screen reader (the Windows test of 28 Nov 2026 found
            // the four boxes unnamed).
            ui.add(
                TextEdit::singleline(&mut self.token)
                    .password(true)
                    .desired_width(f32::INFINITY),
            )
            .labelled_by(words.id);
        }
        if self.checked.as_ref().is_some_and(|c| c.runtime_missing) {
            ui.checkbox(
                &mut self.install_runtime,
                "Fetch and start Microsoft's Visual C++ runtime (checked to be signed by Microsoft; Windows asks for \
                 administrator rights once)",
            );
        }
        let why_not = setup_gate(
            self.checked.as_ref(),
            self.launcher.is_some(),
            self.make_running(),
            self.accepted,
        );
        fixed_row(ui, |ui| {
            if ui
                .add_enabled(why_not.is_none(), Button::new("Set up"))
                .clicked()
            {
                self.start_setup();
            }
            ui.label(why_not.unwrap_or(""));
        });
        ui.separator();
        ui.label("The models and their licences:");
        let mut open: Option<String> = None;
        for licence in licences {
            CollapsingHeader::new(short_name(&licence.name))
                .id_salt(&licence.name)
                .default_open(self.open_folds && licence.full_text_url.is_some())
                .show(ui, |ui| {
                    ui.label(format!("{}: {}", licence.name, licence.job));
                    ui.label(format!(
                        "Licence: {} ({})",
                        licence.licence,
                        progress::gigabytes(licence.bytes)
                    ));
                    if !licence.terms.is_empty() {
                        ui.label(format!("Terms: {}", licence.terms));
                    }
                    if licence.optional {
                        ui.label("Optional: setup installs it unless asked not to.");
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Open the licence page").clicked() {
                            open = Some(licence.licence_url.clone());
                        }
                        if let Some(text) = &licence.full_text_url {
                            if ui.button("Open its full text").clicked() {
                                open = Some(text.clone());
                            }
                        }
                    });
                });
        }
        if let Some(address) = open {
            self.open_address(&address);
        }
    }

    fn start_setup(&mut self) {
        let (Ok(repository), Some(launcher)) = (&self.repository, &self.launcher) else {
            return;
        };
        let token_file = if self.token.trim().is_empty() {
            None
        } else {
            match this_pc::write_token(&self.work, &self.token) {
                Ok(path) => Some(path),
                Err(why) => {
                    self.problem = Some(why);
                    return;
                }
            }
        };
        let choices = Choices {
            licences_accepted: self.accepted,
            install_runtime: self.install_runtime,
            token_file,
        };
        let runtime_step = self.checked.as_ref().is_some_and(|c| c.runtime_missing);
        // Setup::start removes the token's file itself when it cannot start.
        match Setup::start(
            launcher,
            repository,
            &self.work,
            &choices,
            self.models_wanted(),
            runtime_step,
        ) {
            Ok(setup) => {
                self.setup = Some(setup);
                self.token.clear();
                self.problem = None;
            }
            Err(why) => self.problem = Some(why),
        }
    }

    fn setup_rows(&mut self, ui: &mut Ui) {
        let Some(setup) = &self.setup else { return };
        ui.heading("Setting up this PC");
        for (id, state) in &setup.steps {
            fixed_row(ui, |ui| {
                name_cell(ui, step_title(id));
                if *id == "models" {
                    bar(ui, setup.models_bar.shown());
                } else {
                    let done = matches!(state, Step::Done(_) | Step::NotNeeded);
                    bar(ui, if done { 1.0 } else { 0.0 });
                }
                let words = match state {
                    Step::Waiting => "waiting".to_string(),
                    Step::Running(since) => {
                        format!(
                            "{} so far",
                            progress::duration_words(since.elapsed().as_secs_f64())
                        )
                    }
                    Step::Done(took) => {
                        format!("done in {}", progress::duration_words(took.as_secs_f64()))
                    }
                    Step::NotNeeded => "not needed".to_string(),
                    Step::Stopped => "stopped".to_string(),
                };
                ui.label(words);
            });
            if *id == "models" {
                // Always there, so nothing below it moves when the models begin.
                fixed_row(ui, |ui| {
                    name_cell(ui, "");
                    let bytes = format!(
                        "{} of {}",
                        progress::gigabytes(setup.models_have.min(setup.models_wanted)),
                        progress::gigabytes(setup.models_wanted)
                    );
                    ui.label(match (state, setup.models_speed()) {
                        (Step::Running(_), Some((speed, Some(left)))) => format!(
                            "{bytes}; {:.0} MB/s, about {} left",
                            speed / 1e6,
                            progress::duration_words(left)
                        ),
                        (Step::Running(_), _) => format!("{bytes}; speed not known yet"),
                        _ => bytes,
                    });
                });
            }
        }
        let outcome = setup.outcome.clone();
        let lines = setup.running.lines();
        fixed_row(ui, |ui| match &outcome {
            None => {
                if ui.button("Stop setup").clicked() {
                    self.question = Some(Question::StopSetup);
                }
            }
            Some(_) => {
                if ui.button("Back").clicked() {
                    self.setup = None;
                    self.check_again();
                }
            }
        });
        details(ui, &lines, self.open_folds);
        if let Some(outcome) = outcome {
            ui.label(RichText::new(outcome_words(&outcome)).strong());
        }
    }

    // --- Make ---

    fn make_tab(&mut self, ui: &mut Ui) {
        if self.making.is_some() {
            self.make_rows(ui);
            return;
        }
        let why_not = if self.checked.is_none() {
            Some("Checking this PC first…")
        } else if !self.ready() {
            Some("Set this PC up first (This PC).")
        } else if self.setup_running() {
            Some("Setup is running.")
        } else if self.launcher.is_none() {
            Some("model-maker is not beside this program.")
        } else {
            self.form.problem()
        };
        let enabled = self.ready() && !self.setup_running();
        ui.add_enabled_ui(enabled, |ui| {
            let words = ui.label("What to make (one line):");
            ui.add(
                TextEdit::singleline(&mut self.form.prompt)
                    .hint_text("an old looking chair")
                    .char_limit(make::PROMPT_MOST)
                    .desired_width(f32::INFINITY),
            )
            .labelled_by(words.id);
            ui.horizontal(|ui| {
                let words = ui.label("Largest side in metres (empty: usual):");
                ui.add(TextEdit::singleline(&mut self.form.size).desired_width(60.0))
                    .labelled_by(words.id);
            });
            let words = ui.label("The folder it goes in:");
            ui.add(TextEdit::singleline(&mut self.form.out).desired_width(f32::INFINITY))
                .labelled_by(words.id);
            CollapsingHeader::new("More")
                .default_open(self.open_folds)
                .show(ui, |ui| {
                    choice(
                        ui,
                        "Check that it stands on its base",
                        "stands",
                        &mut self.form.stands,
                    );
                    choice(
                        ui,
                        "Check its mirror symmetry",
                        "symmetric",
                        &mut self.form.symmetric,
                    );
                });
        });
        fixed_row(ui, |ui| {
            if ui
                .add_enabled(why_not.is_none(), Button::new("Make"))
                .clicked()
            {
                self.start_make();
            }
        });
        // The reason keeps its own wrapping place below the button.
        ui.label(why_not.unwrap_or(" "));
        ui.label("To look at what you made, or edit it, open the studio.");
    }

    fn start_make(&mut self) {
        let (Ok(repository), Some(launcher)) = (&self.repository, &self.launcher) else {
            return;
        };
        match Making::start(launcher, repository, &self.form) {
            Ok(making) => {
                self.making = Some(making);
                self.problem = None;
            }
            Err(why) => self.problem = Some(why),
        }
    }

    fn make_rows(&mut self, ui: &mut Ui) {
        let Some(making) = &self.making else { return };
        ui.label(format!("Making: {}", self.form.prompt.trim()));
        fixed_row(ui, |ui| {
            name_cell(ui, "Whole make");
            let whole = making.rows.whole.shown();
            bar(ui, whole);
            ui.label(match making.took {
                Some(took) => format!(
                    "{:.0} %, took {}",
                    whole * 100.0,
                    progress::duration_words(took.as_secs_f64())
                ),
                None => format!(
                    "{:.0} %, {} so far",
                    whole * 100.0,
                    progress::duration_words(making.running.started.elapsed().as_secs_f64())
                ),
            });
        });
        let current = making.job.current();
        for (i, stage) in STAGES.iter().enumerate() {
            fixed_row(ui, |ui| {
                name_cell(ui, stage_title(stage));
                bar(ui, making.rows.stage[i].shown());
                ui.label(
                    match (
                        making.job.done[i],
                        current == Some(i) && making.ended.is_none(),
                    ) {
                        (Some(seconds), _) => {
                            format!("done in {}", progress::duration_words(seconds))
                        }
                        (None, _) if making.job.kept(i) => "kept from before".to_string(),
                        (None, true) => {
                            let so_far = making.stage_began.elapsed();
                            format!(
                                "{}; {}",
                                progress::duration_words(so_far.as_secs_f64()),
                                progress::time_left_words(so_far, making.expected[i])
                            )
                        }
                        (None, false) => "waiting".to_string(),
                    },
                );
            });
        }
        let lines = making.running.lines();
        let outcome = making.outcome_words();
        let folder = making.folder().unwrap_or(&making.out).to_path_buf();
        let stopping = making.running.stopping();
        let mut open_folder = false;
        let mut another = false;
        fixed_row(ui, |ui| match &outcome {
            None => {
                if ui.add_enabled(!stopping, Button::new("Cancel")).clicked() {
                    self.question = Some(Question::CancelMake);
                }
                if stopping {
                    ui.label("Stopping…");
                }
            }
            Some(_) => {
                open_folder = ui.button("Open folder").clicked();
                another = ui.button("Make another").clicked();
            }
        });
        details(ui, &lines, self.open_folds);
        if let Some(words) = outcome {
            ui.label(RichText::new(words).strong());
            ui.label(format!("Folder: {}", folder.display()));
            ui.label("To look at it or edit it, open the studio.");
        }
        if open_folder {
            self.open_path(&folder);
        }
        if another {
            self.making = None;
            self.form.prompt.clear();
        }
    }

    fn open_path(&mut self, path: &Path) {
        self.problem = open::that_detached(path)
            .err()
            .map(|e| format!("The folder could not be opened ({e})."));
    }

    /// How long to wait before the next poll.
    pub fn tick(&self) -> Duration {
        if self.busy() || self.checking.is_some() {
            Duration::from_millis(250)
        } else {
            Duration::from_secs(1)
        }
    }
}

fn question_words(question: Question) -> (&'static str, &'static str, &'static str) {
    match question {
        Question::StopSetup => (
            "Stop setup? It carries on where it stopped next time.",
            "Stop setup",
            "Keep going",
        ),
        Question::CancelMake => (
            "Cancel this make? Its folder is kept as it is.",
            "Cancel the make",
            "Keep making",
        ),
        Question::Quit => (
            "Something is running. Quit and stop it?",
            "Stop and quit",
            "Keep working",
        ),
    }
}

pub fn outcome_words(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Done => "Set up: this PC is ready.".to_string(),
        Outcome::Stopped(why) => format!("Setup stopped: {why}"),
        Outcome::EndedUnsaid(code) => format!(
            "Setup ended without saying why (exit code {}); the Details show its output.",
            code.map_or("not known".to_string(), |c| c.to_string())
        ),
        Outcome::Cancelled => {
            "Setup was stopped. It carries on where it stopped next time.".to_string()
        }
    }
}

fn choice(ui: &mut Ui, words: &str, salt: &str, value: &mut Choice) {
    ui.horizontal(|ui| {
        ui.label(words);
        egui::ComboBox::from_id_salt(salt)
            .selected_text(value.words())
            .show_ui(ui, |ui| {
                for option in Choice::ALL {
                    ui.selectable_value(value, option, option.words());
                }
            });
    });
}

fn details(ui: &mut Ui, lines: &[String], open: bool) {
    CollapsingHeader::new("Details (the program's own output)")
        .default_open(open)
        .show(ui, |ui| {
            ScrollArea::vertical()
                .max_height(200.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in lines {
                        ui.label(RichText::new(line).small());
                    }
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_maker_launcher::setup;

    #[test]
    fn every_setup_step_has_plain_words() {
        for (id, _) in setup::STEPS {
            assert_ne!(step_title(id), "Step", "{id}");
        }
        for stage in STAGES {
            assert_eq!(stage_title(stage).to_lowercase(), stage);
        }
    }

    #[test]
    fn only_https_addresses_are_opened() {
        assert!(openable(STUDIO_ADDRESS));
        assert!(openable(
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell"
        ));
        for refused in [
            "http://example.org",
            "file:///etc/passwd",
            "C:\\Windows\\x.exe",
            "https://a b",
            "",
        ] {
            assert!(!openable(refused), "{refused}");
        }
    }

    #[test]
    fn setup_ends_are_said_in_words() {
        assert_eq!(
            outcome_words(&Outcome::Stopped("uv: it failed".into())),
            "Setup stopped: uv: it failed"
        );
        assert!(outcome_words(&Outcome::EndedUnsaid(None)).contains("not known"));
    }

    fn checked(rows: Vec<check::Row>) -> Checked {
        Checked {
            missing: check::missing_count(&rows),
            rows,
            environment_here: false,
            runtime_missing: false,
            setup_done: setup::Done::No,
        }
    }

    fn row(name: &'static str, verdict: Verdict) -> check::Row {
        check::Row {
            check: name,
            found: "x".into(),
            verdict,
        }
    }

    #[test]
    fn set_up_waits_for_the_check_and_the_licences() {
        assert_eq!(
            setup_gate(None, true, false, true),
            Some("Checking this PC first…")
        );
        let fine = checked(vec![row("git", Verdict::Ok("ok".into()))]);
        assert_eq!(
            setup_gate(Some(&fine), true, false, false),
            Some("Accept the licences first.")
        );
        assert_eq!(setup_gate(Some(&fine), true, false, true), None);
        assert!(setup_gate(Some(&fine), false, false, true).is_some());
        assert!(setup_gate(Some(&fine), true, true, true).is_some());
        let missing = checked(vec![row("Memory", Verdict::Missing("16 GB".into()))]);
        assert!(setup_gate(Some(&missing), true, false, true)
            .unwrap()
            .contains("missing"));
        let runtime = checked(vec![row(
            this_pc::RUNTIME_ROW,
            Verdict::Missing("it".into()),
        )]);
        assert_eq!(
            setup_gate(Some(&runtime), true, false, true),
            None,
            "setup can fetch the runtime"
        );
    }

    #[test]
    fn fold_headers_are_short_names() {
        assert_eq!(
            short_name("Qwen 3.5 9B (quantised by bartowski, with its vision part)"),
            "Qwen 3.5 9B"
        );
        assert_eq!(
            short_name("TripoSG (shape), its weights run with our own code"),
            "TripoSG"
        );
        assert_eq!(short_name("SigLIP 2 (So400m, patch 14, 384)"), "SigLIP 2");
        assert_eq!(short_name("FLUX.1 [schnell]"), "FLUX.1 [schnell]");
    }

    #[test]
    fn ok_is_said_once() {
        assert_eq!(
            verdict_words(&row("git", Verdict::Ok("ok".into()))),
            "git: x (ok)"
        );
        assert_eq!(
            verdict_words(&row("Processor", Verdict::Ok("ok (lock linux)".into()))),
            "Processor: x (ok (lock linux))"
        );
        assert_eq!(
            verdict_words(&row("judge", Verdict::Ok("optional".into()))),
            "judge: x (ok: optional)"
        );
        assert_eq!(
            verdict_words(&row("Memory", Verdict::Missing("16 GB".into()))),
            "Memory: x (MISSING: 16 GB)"
        );
    }
}
