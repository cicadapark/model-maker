//! "This PC": the launcher's own check (in this process, as `model-maker check` does it), the models' licences from
//! the registry, and setup run as `model-maker setup` with its steps read from the progress file the launcher writes
//! (`MODEL_MAKER_PROGRESS`), the models' bar from the bytes in the models' folder against the registry's sizes.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use model_maker_launcher::check::{self, Row, Verdict};
use model_maker_launcher::{platform, setup};
use serde::Deserialize;

use crate::children::{self, Running};
use crate::progress::Bar;

// --- the check ---

pub struct Checked {
    pub rows: Vec<Row>,
    pub missing: usize,
    /// The Python environment setup makes is there.
    pub environment_here: bool,
    /// Windows: Microsoft's C++ runtime is missing (setup can fetch it, after asking).
    pub runtime_missing: bool,
    /// Setup's own record (the launcher's check reads it): "ready" is this and the check, nothing of the window's.
    pub setup_done: setup::Done,
}

pub const RUNTIME_ROW: &str = "C++ runtime";

/// The same facts and verdicts as `model-maker check` with no options.
pub fn checked(repository: &Path) -> Checked {
    let place = setup::Place::new(repository, None, None);
    let (system, processor) = platform::here();
    let target = platform::target(system, processor);
    let chosen = target.and_then(|t| platform::chosen_lock(t, None).ok());
    let needs_card = !chosen.is_some_and(|l| l.name.ends_with("-cpu"));
    let facts = check::gather(repository, &place.models);
    let rows = check::rows(&facts, target, chosen, needs_card);
    Checked {
        missing: check::missing_count(&rows),
        environment_here: target.is_some_and(|t| place.venv_python(t).is_file()),
        runtime_missing: rows
            .iter()
            .any(|r| r.check == RUNTIME_ROW && matches!(r.verdict, Verdict::Missing(_))),
        setup_done: facts.setup_done,
        rows,
    }
}

/// Missing things setup cannot help with (all but the C++ runtime, as setup's own `blocking`).
pub fn blocking(rows: &[Row]) -> usize {
    rows.iter()
        .filter(|r| matches!(r.verdict, Verdict::Missing(_)) && r.check != RUNTIME_ROW)
        .count()
}

pub fn folder_bytes(folder: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => folder_bytes(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

// --- the licences ---

#[derive(Deserialize)]
struct Registry {
    model: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    name: String,
    job: String,
    licence: String,
    licence_url: String,
    #[serde(default)]
    conditions: String,
    #[serde(default)]
    gated: bool,
    optional: Option<String>,
    #[serde(default)]
    parts: Vec<Part>,
    licence_text: Option<Text>,
}

#[derive(Deserialize)]
struct Part {
    #[serde(default)]
    size_mb: u64,
    fetch: Option<bool>,
}

#[derive(Deserialize)]
struct Text {
    url: String,
}

/// One model as the licence list shows it (fetch_models.py shows the same fields before its question).
#[derive(Clone, Debug, PartialEq)]
pub struct Licence {
    pub name: String,
    pub job: String,
    pub licence: String,
    pub licence_url: String,
    pub terms: String,
    pub full_text_url: Option<String>,
    pub gated: bool,
    pub optional: bool,
    pub bytes: u64,
}

pub fn licences(registry: &str) -> Result<Vec<Licence>, String> {
    let read: Registry =
        toml::from_str(registry).map_err(|e| format!("registry/models.toml: {e}"))?;
    Ok(read
        .model
        .into_iter()
        .map(|m| Licence {
            bytes: m
                .parts
                .iter()
                .filter(|p| p.fetch != Some(false))
                .map(|p| p.size_mb * 1_000_000)
                .sum(),
            name: m.name,
            job: m.job,
            licence: m.licence,
            licence_url: m.licence_url,
            terms: m.conditions,
            full_text_url: m.licence_text.map(|t| t.url),
            gated: m.gated,
            optional: m.optional.is_some(),
        })
        .collect())
}

pub fn read_licences(repository: &Path) -> Result<Vec<Licence>, String> {
    let path = repository.join("registry").join("models.toml");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    licences(&text)
}

/// The models' bytes setup fetches (the optional question judge too: setup's default), for the models' bar only.
pub fn bytes_wanted(licences: &[Licence]) -> u64 {
    licences.iter().map(|l| l.bytes).sum()
}

/// This PC is ready for a make, as the launcher says it: its check passes, and setup's own record says it ended well
/// with the models. The window adds no rule of its own.
pub fn ready(checked: &Checked) -> bool {
    checked.missing == 0
        && checked.environment_here
        && checked.setup_done == setup::Done::WithModels
}

// --- setup ---

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Began(String),
    Done,
    Stopped(String),
}

/// The progress file's lines (whole lines only; one of another schema is skipped).
pub fn events(text: &str) -> Vec<Event> {
    let whole = match text.rfind('\n') {
        Some(end) => &text[..end],
        None => "",
    };
    whole
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| v["schema"].as_u64() == Some(u64::from(setup::PROGRESS_SCHEMA)))
        .filter_map(|v| match v["event"].as_str()? {
            "began" => Some(Event::Began(v["step"].as_str()?.to_string())),
            "done" => Some(Event::Done),
            "stopped" => Some(Event::Stopped(v["why"].as_str().unwrap_or("").to_string())),
            _ => None,
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Waiting,
    Running(Instant),
    Done(Duration),
    /// Setup ended without this step (not needed on this PC, or skipped).
    NotNeeded,
    Stopped,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Done,
    Stopped(String),
    /// The program ended without its last line (killed, or it could not start).
    EndedUnsaid(Option<i32>),
    Cancelled,
}

/// The steps' states after these events, seen at `now` (each step's start is when its line was first seen).
pub fn apply(steps: &mut [(&'static str, Step)], events: &[Event], now: Instant) {
    let finish_running = |steps: &mut [(&'static str, Step)], as_done: bool| {
        for (_, state) in steps.iter_mut() {
            if let Step::Running(since) = state {
                *state = if as_done {
                    Step::Done(now - *since)
                } else {
                    Step::Stopped
                };
            }
        }
    };
    for event in events {
        match event {
            Event::Began(id) => {
                if steps
                    .iter()
                    .any(|(known, state)| known == id && *state != Step::Waiting)
                {
                    continue;
                }
                finish_running(steps, true);
                if let Some((_, state)) = steps.iter_mut().find(|(known, _)| known == id) {
                    *state = Step::Running(now);
                }
            }
            Event::Done => {
                finish_running(steps, true);
                for (_, state) in steps.iter_mut() {
                    if *state == Step::Waiting {
                        *state = Step::NotNeeded;
                    }
                }
            }
            Event::Stopped(_) => finish_running(steps, false),
        }
    }
}

pub fn outcome(events: &[Event]) -> Option<Outcome> {
    events.iter().rev().find_map(|e| match e {
        Event::Done => Some(Outcome::Done),
        Event::Stopped(why) => Some(Outcome::Stopped(why.clone())),
        Event::Began(_) => None,
    })
}

/// What the person chose in the window before "Set up".
#[derive(Clone, Debug, Default)]
pub struct Choices {
    pub licences_accepted: bool,
    pub install_runtime: bool,
    pub token_file: Option<PathBuf>,
}

/// `model-maker setup`'s words for these choices: nothing the command line does not have.
pub fn setup_words(choices: &Choices) -> Vec<OsString> {
    let mut words: Vec<OsString> = vec!["setup".into()];
    if choices.licences_accepted {
        words.push("--yes".into());
    }
    if choices.install_runtime {
        words.push("--install-cxx-runtime".into());
    }
    if let Some(token) = &choices.token_file {
        words.push("--hf-token-file".into());
        words.push(token.clone().into());
    }
    words
}

/// The hub token's file in the window's own folder (see `own_folder`).
pub const TOKEN_FILE: &str = "hub-token";

/// The window's own folder: inside the repository's tools folder (the person's own, never a shared temporary
/// folder), only this user's on Unix. A hub token left by a crash is removed once this window holds the lock
/// (`remove_token`), never under a window that is still open.
pub fn own_folder(repository: &Path) -> Result<PathBuf, String> {
    let folder = setup::Place::new(repository, None, None)
        .tools
        .join("companion");
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("{}: {e}", folder.display()))?;
    }
    Ok(folder)
}

pub fn remove_token(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// The hub token, for the time setup runs only: a new file only this user can read (an existing file is refused,
/// never written into), removed when setup ends, when it could not start, and when the window closes.
pub fn write_token(folder: &Path, token: &str) -> Result<PathBuf, String> {
    let path = folder.join(TOKEN_FILE);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    use std::io::Write;
    options
        .open(&path)
        .and_then(|mut f| f.write_all(token.trim().as_bytes()))
        .map_err(|e| format!("the token could not be written for setup ({e})"))?;
    Ok(path)
}

pub struct Setup {
    pub running: Running,
    progress_file: PathBuf,
    token_file: Option<PathBuf>,
    pub steps: Vec<(&'static str, Step)>,
    pub outcome: Option<Outcome>,
    pub models_bar: Bar,
    models_folder: PathBuf,
    pub models_have: u64,
    pub models_wanted: u64,
    models_at_start: u64,
    models_began: Option<Instant>,
    last_size_read: Option<Instant>,
}

/// How often the models' folder is measured.
pub const MODELS_READ_EVERY: Duration = Duration::from_secs(2);

impl Drop for Setup {
    /// The token never outlives the window (closed during setup) or a setup that has ended.
    fn drop(&mut self) {
        if let Some(token) = self.token_file.take() {
            remove_token(&token);
        }
    }
}

impl Setup {
    fn started(
        launcher: &Path,
        work: &Path,
        choices: &Choices,
    ) -> Result<(Running, PathBuf), String> {
        std::fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
        let progress_file = work.join("setup-progress.jsonl");
        let _ = std::fs::remove_file(&progress_file);
        let settings = [(
            OsString::from(setup::PROGRESS_SETTING),
            progress_file.clone().into_os_string(),
        )];
        let running = children::start(launcher, &setup_words(choices), &settings)?;
        Ok((running, progress_file))
    }

    pub fn start(
        launcher: &Path,
        repository: &Path,
        work: &Path,
        choices: &Choices,
        models_wanted: u64,
        with_runtime_step: bool,
    ) -> Result<Setup, String> {
        let started = Setup::started(launcher, work, choices);
        if started.is_err() {
            if let Some(token) = &choices.token_file {
                remove_token(token);
            }
        }
        let (running, progress_file) = started?;
        let models_folder = setup::Place::new(repository, None, None).models;
        let models_at_start = folder_bytes(&models_folder);
        Ok(Setup {
            running,
            progress_file,
            token_file: choices.token_file.clone(),
            steps: setup::STEPS
                .iter()
                .filter(|(id, _)| with_runtime_step || *id != "runtime")
                .map(|(id, _)| (*id, Step::Waiting))
                .collect(),
            outcome: None,
            models_bar: Bar::default(),
            models_have: models_at_start,
            models_at_start,
            models_wanted,
            models_folder,
            models_began: None,
            last_size_read: None,
        })
    }

    /// Reads what changed; true while it runs.
    pub fn poll(&mut self) -> bool {
        if self.outcome.is_some() {
            return false;
        }
        let ended = self.running.ended();
        let text = std::fs::read_to_string(&self.progress_file).unwrap_or_default();
        let events = events(&text);
        let now = Instant::now();
        let mut fresh: Vec<(&'static str, Step)> = self
            .steps
            .iter()
            .map(|(id, _)| (*id, Step::Waiting))
            .collect();
        apply(&mut fresh, &events, now);
        // Keep each step's first-seen start: a step's time is from when its line first appeared.
        for ((_, old), (_, new)) in self.steps.iter().zip(fresh.iter_mut()) {
            if let (Step::Running(first), Step::Running(_)) = (old, &*new) {
                *new = Step::Running(*first);
            }
            if let (Step::Running(first), Step::Done(_)) = (old, &*new) {
                *new = Step::Done(now - *first);
            }
            if let Step::Done(took) = old {
                *new = Step::Done(*took);
            }
        }
        self.steps = fresh;
        self.read_models(now);
        if let Some(ended) = ended {
            self.outcome = Some(if ended.stopped_by_us {
                Outcome::Cancelled
            } else {
                outcome(&events).unwrap_or(Outcome::EndedUnsaid(ended.code))
            });
            if !matches!(self.outcome, Some(Outcome::Done)) {
                for (_, state) in self.steps.iter_mut() {
                    if matches!(state, Step::Running(_)) {
                        *state = Step::Stopped;
                    }
                }
            }
            if let Some(token) = self.token_file.take() {
                remove_token(&token);
            }
            return false;
        }
        true
    }

    fn read_models(&mut self, now: Instant) {
        let models_running = self
            .steps
            .iter()
            .any(|(id, s)| *id == "models" && matches!(s, Step::Running(_)));
        let models_done = self
            .steps
            .iter()
            .any(|(id, s)| *id == "models" && matches!(s, Step::Done(_)));
        if models_running && self.models_began.is_none() {
            self.models_began = Some(now);
            self.models_at_start = folder_bytes(&self.models_folder);
        }
        if (models_running || models_done)
            && self
                .last_size_read
                .is_none_or(|t| now - t >= MODELS_READ_EVERY)
        {
            self.models_have = folder_bytes(&self.models_folder);
            self.last_size_read = Some(now);
        }
        if models_done {
            self.models_bar.to(1.0);
        } else if self.models_wanted > 0 {
            self.models_bar
                .to(self.models_have as f32 / self.models_wanted as f32);
        }
    }

    /// Bytes a second since the models' step began, and the time left at that speed.
    pub fn models_speed(&self) -> Option<(f64, Option<f64>)> {
        let began = self.models_began?;
        let seconds = began.elapsed().as_secs_f64();
        let fetched = self.models_have.saturating_sub(self.models_at_start) as f64;
        if seconds < 5.0 || fetched <= 0.0 {
            return None;
        }
        let speed = fetched / seconds;
        let left = self.models_wanted.saturating_sub(self.models_have) as f64;
        Some((speed, (speed > 0.0).then(|| left / speed)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTRY: &str = r#"
format = 1
[[model]]
id = "a"
name = "Model A"
role = "picture"
job = "Paints."
version = "1"
licence = "Apache-2.0"
licence_url = "https://example.org/a"
conditions = "Weights Apache-2.0."
gated = true
parts = [ { url = "u1", sha256 = "s", size_mb = 100 }, { url = "u2", sha256 = "s", size_mb = 5, fetch = false } ]

[[model]]
id = "b"
name = "Model B"
role = "judge"
job = "Asks."
version = "2"
licence = "CreativeML Open RAIL++-M"
licence_url = "https://example.org/b"
licence_text = { url = "https://example.org/b/LICENSE.md", sha256 = "x" }
optional = "question-judge"
parts = [ { url = "u3", sha256 = "s", size_mb = 7 } ]
"#;

    #[test]
    fn the_licences_are_read_with_their_terms_and_sizes() {
        let read = licences(REGISTRY).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].name, "Model A");
        assert!(read[0].gated && !read[0].optional);
        assert_eq!(
            read[0].bytes, 100_000_000,
            "a part not fetched is not counted"
        );
        assert_eq!(
            read[1].full_text_url.as_deref(),
            Some("https://example.org/b/LICENSE.md")
        );
        assert_eq!(bytes_wanted(&read), 107_000_000);
        assert!(licences("model = 3").is_err());
    }

    #[test]
    fn the_real_registry_is_read() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let read = read_licences(&repository).unwrap();
        assert!(read.len() >= 5);
        assert!(read
            .iter()
            .all(|l| !l.licence.is_empty() && l.licence_url.starts_with("https://")));
        let all = bytes_wanted(&read);
        assert!(all > 40_000_000_000, "{all}");
    }

    #[test]
    fn progress_lines_become_steps() {
        let file = concat!(
            "{\"schema\":1,\"event\":\"began\",\"step\":\"python\"}\n",
            "{\"schema\":9,\"event\":\"began\",\"step\":\"tools\"}\n",
            "not json\n",
            "{\"schema\":1,\"event\":\"began\",\"step\":\"models\"}\n",
            "{\"schema\":1,\"event\":\"done\"}"
        );
        let read = events(file);
        assert_eq!(
            read,
            [Event::Began("python".into()), Event::Began("models".into())],
            "a cut-off last line waits"
        );
        let now = Instant::now();
        let mut steps: Vec<(&'static str, Step)> = ["python", "tools", "models", "self-test"]
            .map(|id| (id, Step::Waiting))
            .to_vec();
        apply(&mut steps, &read, now);
        assert!(matches!(steps[0].1, Step::Done(_)));
        assert_eq!(steps[1].1, Step::Waiting);
        assert_eq!(steps[2].1, Step::Running(now));
        let mut all = read.clone();
        all.push(Event::Done);
        apply(&mut steps, &[Event::Done], now);
        assert!(matches!(steps[2].1, Step::Done(_)));
        assert_eq!(steps[1].1, Step::NotNeeded);
        assert_eq!(outcome(&all), Some(Outcome::Done));
    }

    #[test]
    fn a_stopped_setup_says_why_and_stops_its_step() {
        let read = events("{\"schema\":1,\"event\":\"began\",\"step\":\"python\"}\n{\"schema\":1,\"event\":\"stopped\",\"why\":\"uv: it failed\"}\n");
        let mut steps: Vec<(&'static str, Step)> =
            vec![("python", Step::Waiting), ("tools", Step::Waiting)];
        apply(&mut steps, &read, Instant::now());
        assert_eq!(steps[0].1, Step::Stopped);
        assert_eq!(steps[1].1, Step::Waiting);
        assert_eq!(
            outcome(&read),
            Some(Outcome::Stopped("uv: it failed".into()))
        );
        assert_eq!(outcome(&read[..1]), None);
    }

    #[test]
    fn setup_gets_only_the_command_lines_own_words() {
        let none: Vec<String> = setup_words(&Choices::default())
            .iter()
            .map(|w| w.to_string_lossy().into())
            .collect();
        assert_eq!(none, ["setup"]);
        let all = setup_words(&Choices {
            licences_accepted: true,
            install_runtime: true,
            token_file: Some(PathBuf::from("/t/hub-token")),
        });
        let all: Vec<String> = all.iter().map(|w| w.to_string_lossy().into()).collect();
        assert_eq!(
            all,
            [
                "setup",
                "--yes",
                "--install-cxx-runtime",
                "--hf-token-file",
                "/t/hub-token"
            ]
        );
    }

    #[test]
    fn the_token_file_is_new_the_users_own_and_never_left_behind() {
        let repository =
            std::env::temp_dir().join(format!("companion-token-{}", std::process::id()));
        let folder = own_folder(&repository).unwrap();
        assert!(
            folder.starts_with(repository.join(".tools")),
            "the repository's own tools folder"
        );
        let path = write_token(&folder, "  hf_abc \n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hf_abc");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&path), 0o600);
            assert_eq!(mode(&folder), 0o700);
        }
        assert!(
            write_token(&folder, "other").is_err(),
            "a file already there is refused, not written into"
        );
        remove_token(&own_folder(&repository).unwrap().join(TOKEN_FILE));
        assert!(
            !path.exists(),
            "a token left by a crash is removed at the next start"
        );
        // A setup that cannot start removes its token at once.
        let path = write_token(&folder, "hf_abc").unwrap();
        let choices = Choices {
            token_file: Some(path.clone()),
            ..Choices::default()
        };
        let refused = Setup::start(
            Path::new("/no/such/launcher"),
            &repository,
            &folder,
            &choices,
            1,
            false,
        );
        assert!(refused.is_err());
        assert!(!path.exists(), "removed when setup could not start");
        std::fs::remove_dir_all(&repository).unwrap();
    }

    #[test]
    fn ready_is_the_launchers_check_and_setups_own_record() {
        let mut checked = Checked {
            rows: vec![],
            missing: 0,
            environment_here: true,
            runtime_missing: false,
            setup_done: setup::Done::WithModels,
        };
        assert!(ready(&checked));
        checked.setup_done = setup::Done::WithoutModels;
        assert!(!ready(&checked), "set up without the models: no make yet");
        checked.setup_done = setup::Done::No;
        assert!(
            !ready(&checked),
            "a stopped setup is never ready, whatever the folder holds"
        );
        checked.setup_done = setup::Done::WithModels;
        checked.environment_here = false;
        assert!(!ready(&checked));
        checked.environment_here = true;
        checked.missing = 1;
        assert!(!ready(&checked));
    }

    #[test]
    fn only_the_runtime_does_not_block() {
        let row = |check: &'static str| Row {
            check,
            found: String::new(),
            verdict: Verdict::Missing("x".into()),
        };
        assert_eq!(blocking(&[row(RUNTIME_ROW)]), 0);
        assert_eq!(blocking(&[row(RUNTIME_ROW), row("Memory")]), 1);
    }
}

/// Setup through a stand-in launcher that writes the progress file as the launcher does.
#[cfg(all(test, unix))]
mod through_a_stand_in {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    const STAND_IN: &str = r#"#!/bin/sh
p="$MODEL_MAKER_PROGRESS"
echo '{"schema":1,"event":"began","step":"python"}' >> "$p"
echo "installing"
echo '{"schema":1,"event":"began","step":"models"}' >> "$p"
if [ "$2" != "--yes" ]; then
  echo '{"schema":1,"event":"stopped","why":"no licence was accepted"}' >> "$p"
  exit 1
fi
echo '{"schema":1,"event":"done"}' >> "$p"
"#;

    fn run(name: &str, choices: &Choices) -> Setup {
        let root =
            std::env::temp_dir().join(format!("companion-setup-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("repo")).unwrap();
        let script = root.join("model-maker");
        std::fs::write(&script, STAND_IN).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut setup = Setup::start(
            &script,
            &root.join("repo"),
            &root.join("work"),
            choices,
            1000,
            false,
        )
        .unwrap();
        let since = Instant::now();
        while setup.poll() && since.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::fs::remove_dir_all(&root).unwrap();
        setup
    }

    #[test]
    fn a_whole_setup_ends_done_with_its_steps() {
        let setup = run(
            "done",
            &Choices {
                licences_accepted: true,
                ..Choices::default()
            },
        );
        assert_eq!(setup.outcome, Some(Outcome::Done));
        let states: Vec<&Step> = setup.steps.iter().map(|(_, s)| s).collect();
        assert!(
            matches!(states[0], Step::Done(_)),
            "python: {:?}",
            states[0]
        );
        assert_eq!(*states[1], Step::NotNeeded, "tools never began");
        assert!(matches!(states[2], Step::Done(_)), "models");
        assert_eq!(setup.models_bar.shown(), 1.0);
        assert!(setup.running.lines().contains(&"installing".to_string()));
    }

    #[test]
    fn closing_during_setup_removes_the_token() {
        let root =
            std::env::temp_dir().join(format!("companion-setup-drop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("repo")).unwrap();
        let script = root.join("model-maker");
        std::fs::write(&script, "#!/bin/sh\nexec sleep 30\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let folder = own_folder(&root.join("repo")).unwrap();
        let token = write_token(&folder, "hf_abc").unwrap();
        let choices = Choices {
            token_file: Some(token.clone()),
            ..Choices::default()
        };
        let setup = Setup::start(&script, &root.join("repo"), &folder, &choices, 1, false).unwrap();
        assert!(token.exists());
        drop(setup);
        assert!(!token.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_stopped_setup_shows_the_launchers_own_why() {
        let setup = run("stopped", &Choices::default());
        assert_eq!(
            setup.outcome,
            Some(Outcome::Stopped("no licence was accepted".into()))
        );
        assert_eq!(setup.steps[2].1, Step::Stopped);
    }
}
