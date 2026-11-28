//! "Make": one object from a prompt, as `model-maker make` with exactly the person's choices, followed through the job
//! ledger's lines (whole lines only, nothing the program prints is read for meaning), and its result line from the
//! job's sidecar as written (plan sections 1.4 and 4). Leaving the screen forgets the make.

use std::ffi::OsString;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::progress;

/// The pipeline's four stages, in order (`--stages`' default).
pub const STAGES: [&str; 4] = ["pictures", "shape", "paint", "finish"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Auto,
    Yes,
    No,
}

impl Choice {
    pub const ALL: [Choice; 3] = [Choice::Auto, Choice::Yes, Choice::No];

    pub fn words(self) -> &'static str {
        match self {
            Choice::Auto => "auto",
            Choice::Yes => "yes",
            Choice::No => "no",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Form {
    pub prompt: String,
    /// Metres, as typed; empty is the pipeline's own choice.
    pub size: String,
    pub out: String,
    pub stands: Choice,
    pub symmetric: Choice,
}

/// Where made objects go unless the person picks another folder.
pub fn default_out() -> PathBuf {
    std::env::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("model-maker")
}

impl Default for Form {
    fn default() -> Form {
        Form {
            prompt: String::new(),
            size: String::new(),
            out: default_out().display().to_string(),
            stands: Choice::Auto,
            symmetric: Choice::Auto,
        }
    }
}

/// The longest prompt the box takes (a prompt is a line, not a page).
pub const PROMPT_MOST: usize = 200;

impl Form {
    fn size_metres(&self) -> Result<Option<f64>, &'static str> {
        let typed = self.size.trim().replace(',', ".");
        if typed.is_empty() {
            return Ok(None);
        }
        match typed.parse::<f64>() {
            Ok(metres) if metres.is_finite() && metres > 0.0 && metres <= 100.0 => Ok(Some(metres)),
            _ => Err("The size is a number of metres, such as 0.9."),
        }
    }

    /// Why "Make" cannot start yet, in words; None when it can.
    pub fn problem(&self) -> Option<&'static str> {
        let prompt = self.prompt.trim();
        if prompt.is_empty() {
            return Some("Write what to make, such as \"an old looking chair\".");
        }
        if prompt.chars().count() > PROMPT_MOST {
            return Some("The prompt is too long: one line of up to 200 letters.");
        }
        if prompt.chars().any(char::is_control) {
            return Some("The prompt is one line.");
        }
        if prompt.starts_with('-') {
            // The command would read it as one of its options, not as the prompt.
            return Some("The prompt cannot start with a dash.");
        }
        if self.out.trim().is_empty() {
            return Some("Choose the folder the object goes in.");
        }
        self.size_metres().err()
    }

    /// `model-maker make`'s words: exactly these choices, nothing else.
    pub fn words(&self) -> Vec<OsString> {
        let mut words: Vec<OsString> = vec![
            "make".into(),
            self.prompt.trim().into(),
            "--out".into(),
            self.out.trim().into(),
        ];
        if let Ok(Some(metres)) = self.size_metres() {
            words.push("--size".into());
            words.push(metres.to_string().into());
        }
        for (flag, choice) in [("--stands", self.stands), ("--symmetric", self.symmetric)] {
            if choice != Choice::Auto {
                words.push(flag.into());
                words.push(choice.words().into());
            }
        }
        words
    }
}

// --- the ledger ---

/// The ledger the pipeline writes (ledger.py's `folder`): MODEL_MAKER_LEDGER, else the repository's metrics/.
pub fn ledger_path(repository: &Path) -> PathBuf {
    std::env::var_os("MODEL_MAKER_LEDGER")
        .map(PathBuf::from)
        .unwrap_or_else(|| repository.join("metrics"))
        .join("jobs.jsonl")
}

/// Reads the lines added to a file since it was opened, whole lines only.
pub struct Follower {
    path: PathBuf,
    offset: u64,
    partial: Vec<u8>,
}

impl Follower {
    /// From the file's present end: only lines written after this.
    pub fn from_end(path: &Path) -> Follower {
        let offset = std::fs::metadata(path).map_or(0, |m| m.len());
        Follower {
            path: path.to_path_buf(),
            offset,
            partial: Vec::new(),
        }
    }

    pub fn new_lines(&mut self) -> Vec<Value> {
        let Ok(mut file) = std::fs::File::open(&self.path) else {
            return Vec::new();
        };
        let length = file.metadata().map_or(0, |m| m.len());
        if length < self.offset {
            // The file was replaced: start again from its beginning.
            self.offset = 0;
            self.partial.clear();
        }
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let mut added = Vec::new();
        if file.read_to_end(&mut added).is_err() {
            return Vec::new();
        }
        self.offset += added.len() as u64;
        self.partial.extend_from_slice(&added);
        let Some(end) = self.partial.iter().rposition(|b| *b == b'\n') else {
            return Vec::new();
        };
        let whole: Vec<u8> = self.partial.drain(..=end).collect();
        String::from_utf8_lossy(&whole)
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

/// A line of this make: its out folder's name as batch, and (when it has one) its job folder inside that folder.
pub fn is_ours(line: &Value, out: &Path) -> bool {
    let batch = out.file_name().map(|n| n.to_string_lossy().into_owned());
    if line["batch"].as_str() != batch.as_deref() {
        return false;
    }
    match line["folder"].as_str() {
        None => true,
        Some(folder) => Path::new(folder).parent() == Some(out),
    }
}

/// The median of this PC's own past times of each stage on this card (the ledger's "stage" lines).
pub fn expected_seconds(ledger: &str, card: Option<&str>) -> [Option<f64>; 4] {
    let lines: Vec<Value> = ledger
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    STAGES.map(|stage| {
        progress::median(
            lines
                .iter()
                .filter(|l| l["event"] == "stage" && l["stage"] == stage)
                .filter(|l| card.is_none_or(|c| l["machine"]["card"].as_str() == Some(c)))
                .filter_map(|l| l["seconds"].as_f64())
                .collect(),
        )
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum End {
    /// The job's own line: its folder and name, for the sidecar.
    Made {
        folder: PathBuf,
        slug: String,
        passed: bool,
        failed: Vec<String>,
    },
    NotMade(String),
    Aborted {
        stage: String,
        error: String,
    },
}

/// What the ledger said of this make so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Job {
    /// Each stage's seconds once it is done.
    pub done: [Option<f64>; 4],
    pub end: Option<End>,
    /// The card it runs on (from any of its lines), for the expected times.
    pub card: Option<String>,
}

impl Job {
    pub fn apply(&mut self, line: &Value) {
        if let Some(card) = line["machine"]["card"].as_str() {
            self.card.get_or_insert_with(|| card.to_string());
        }
        let text = |key: &str| line[key].as_str().unwrap_or("").to_string();
        match line["event"].as_str() {
            Some("stage") => {
                if let Some(i) = STAGES.iter().position(|s| line["stage"] == *s) {
                    self.done[i] = Some(line["seconds"].as_f64().unwrap_or(0.0));
                }
            }
            Some("aborted") => {
                self.end = Some(End::Aborted {
                    stage: text("stage"),
                    error: text("error"),
                })
            }
            Some("not made") => self.end = Some(End::NotMade(text("why"))),
            Some("job") => {
                self.end = Some(End::Made {
                    folder: PathBuf::from(text("folder")),
                    slug: text("slug"),
                    passed: line["passed"].as_bool().unwrap_or(false),
                    failed: line["failed"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|n| n.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                })
            }
            _ => {}
        }
    }

    /// The stage under way: the first not done after the last one done (a stage before a done one was kept).
    pub fn current(&self) -> Option<usize> {
        let after = self
            .done
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |last| last + 1);
        (after < STAGES.len()).then_some(after)
    }

    /// A stage with no line in this make although a later stage has one, or the make ended made: the pipeline kept
    /// its result from an earlier make into the same folder ("Make another" with the same words).
    pub fn kept(&self, stage: usize) -> bool {
        self.done[stage].is_none()
            && (matches!(self.end, Some(End::Made { .. }))
                || self.done[stage + 1..].iter().any(Option::is_some))
    }
}

/// The result line from the sidecar's own counts: its counted checks, and those that failed.
pub fn result_words(sidecar: &Value) -> Option<String> {
    let groups = sidecar["checks"].as_object()?;
    let counted: Vec<&Value> = groups
        .values()
        .filter_map(Value::as_array)
        .flatten()
        .filter(|c| c["counted"].as_bool() != Some(false))
        .collect();
    let failed: Vec<&str> = counted
        .iter()
        .filter(|c| c["passed"].as_bool() != Some(true))
        .map(|c| c["name"].as_str().unwrap_or("a check"))
        .collect();
    Some(match failed.len() {
        0 => format!("Made: passes {} of {} checks", counted.len(), counted.len()),
        1 => format!(
            "Made, with 1 failed check of {}: {}",
            counted.len(),
            failed[0]
        ),
        n => format!(
            "Made, with {n} failed checks of {}: {}",
            counted.len(),
            failed.join("; ")
        ),
    })
}

/// The words for a make's end.
pub fn end_words(end: &End) -> String {
    match end {
        End::Made {
            folder,
            slug,
            passed,
            failed,
        } => {
            let sidecar = folder.join(format!("{slug}.provenance.json"));
            std::fs::read_to_string(&sidecar)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .and_then(|v| result_words(&v))
                .unwrap_or_else(|| match (passed, failed.len()) {
                    (true, _) => "Made: passes every check".to_string(),
                    (false, 1) => format!("Made, with 1 failed check: {}", failed[0]),
                    (false, n) => format!("Made, with {n} failed checks: {}", failed.join("; ")),
                })
        }
        End::NotMade(why) => format!("Not made: {why}"),
        End::Aborted { stage, error } => format!("Stopped in {stage}: {error}"),
    }
}

/// The bars of one stage row.
#[derive(Clone, Debug, Default)]
pub struct Rows {
    pub stage: [progress::Bar; 4],
    pub whole: progress::Bar,
}

/// Moves the bars for the job at `elapsed_in_stage` into its current stage.
pub fn advance(
    rows: &mut Rows,
    job: &Job,
    expected: &[Option<f64>; 4],
    elapsed_in_stage: Duration,
) {
    for i in 0..STAGES.len() {
        if job.done[i].is_some() || job.kept(i) {
            rows.stage[i].to(1.0);
        }
    }
    if job.end.is_none() {
        if let Some(i) = job.current() {
            if let Some(expected) = expected[i] {
                rows.stage[i].to(progress::by_clock(elapsed_in_stage, expected));
            }
        }
    }
    let sum: f32 = rows.stage.iter().map(progress::Bar::shown).sum();
    rows.whole.to(sum / STAGES.len() as f32);
}

/// The make under way, or ended and shown until the person leaves the screen.
pub struct Making {
    pub running: crate::children::Running,
    pub out: PathBuf,
    follower: Follower,
    pub job: Job,
    pub rows: Rows,
    pub expected: [Option<f64>; 4],
    history: String,
    /// When the stage under way began (the end of the one before).
    pub stage_began: Instant,
    pub ended: Option<crate::children::Ended>,
    pub left_running: Option<bool>,
    /// The whole make's time, once it ended.
    pub took: Option<Duration>,
}

impl Making {
    pub fn start(launcher: &Path, repository: &Path, form: &Form) -> Result<Making, String> {
        let out = PathBuf::from(form.out.trim());
        std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        // Plain C:\... as the pipeline writes it in the ledger (model_maker_launcher::paths).
        let out = model_maker_launcher::paths::real(&out);
        let ledger = ledger_path(repository);
        let history = std::fs::read_to_string(&ledger).unwrap_or_default();
        let follower = Follower::from_end(&ledger);
        let mut words = form.words();
        words[3] = out.clone().into_os_string();
        let running = crate::children::start(launcher, &words, &[])?;
        Ok(Making {
            running,
            out,
            follower,
            job: Job::default(),
            rows: Rows::default(),
            expected: expected_seconds(&history, None),
            history,
            stage_began: Instant::now(),
            ended: None,
            left_running: None,
            took: None,
        })
    }

    /// Reads what changed; true while it runs.
    pub fn poll(&mut self) -> bool {
        let ended = if self.ended.is_none() {
            self.running.ended()
        } else {
            None
        };
        for line in self.follower.new_lines() {
            if !is_ours(&line, &self.out) {
                continue;
            }
            let before = self.job.done.iter().filter(|d| d.is_some()).count();
            let had_card = self.job.card.is_some();
            self.job.apply(&line);
            if self.job.done.iter().filter(|d| d.is_some()).count() > before {
                self.stage_began = Instant::now();
            }
            if !had_card && self.job.card.is_some() {
                self.expected = expected_seconds(&self.history, self.job.card.as_deref());
            }
        }
        advance(
            &mut self.rows,
            &self.job,
            &self.expected,
            self.stage_began.elapsed(),
        );
        if let Some(ended) = ended {
            self.ended = Some(ended);
            self.left_running = Some(self.running.anything_left());
            self.took = Some(self.running.started.elapsed());
        }
        self.ended.is_none()
    }

    /// The make's last line, once it has ended.
    pub fn outcome_words(&self) -> Option<String> {
        let ended = self.ended?;
        if ended.stopped_by_us {
            // Only Unix counts what is left (its process group); on Windows the launcher's job object ends what
            // the make started, which is not counted here, so nothing is claimed.
            if cfg!(windows) {
                return Some("Cancelled. Its folder is kept as it is.".to_string());
            }
            let left = match self.left_running {
                Some(false) => "nothing of it is left running",
                _ => "something of it may still be running",
            };
            return Some(format!("Cancelled: {left}. Its folder is kept as it is."));
        }
        Some(match &self.job.end {
            Some(end) => end_words(end),
            None => format!(
                "The make ended without a result (exit code {}); the Details show why.",
                ended
                    .code
                    .map_or("not known".to_string(), |c| c.to_string())
            ),
        })
    }

    /// The job's folder, once the ledger named it.
    pub fn folder(&self) -> Option<&Path> {
        match &self.job.end {
            Some(End::Made { folder, .. }) => Some(folder),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn strings(words: Vec<OsString>) -> Vec<String> {
        words
            .iter()
            .map(|w| w.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn the_form_gives_exactly_its_choices() {
        let mut form = Form {
            prompt: "  an old looking chair ".into(),
            out: "/somewhere/made".into(),
            ..Form::default()
        };
        assert_eq!(form.problem(), None);
        assert_eq!(
            strings(form.words()),
            ["make", "an old looking chair", "--out", "/somewhere/made"]
        );
        form.size = "0,9".into();
        form.stands = Choice::Yes;
        form.symmetric = Choice::No;
        assert_eq!(
            strings(form.words()),
            [
                "make",
                "an old looking chair",
                "--out",
                "/somewhere/made",
                "--size",
                "0.9",
                "--stands",
                "yes",
                "--symmetric",
                "no"
            ]
        );
    }

    #[test]
    fn a_form_that_cannot_start_says_why() {
        let form = |prompt: &str, size: &str, out: &str| Form {
            prompt: prompt.into(),
            size: size.into(),
            out: out.into(),
            ..Form::default()
        };
        assert!(form("", "", "/o")
            .problem()
            .unwrap()
            .contains("Write what to make"));
        assert!(form("a\nb", "", "/o")
            .problem()
            .unwrap()
            .contains("one line"));
        assert!(form(&"x".repeat(201), "", "/o")
            .problem()
            .unwrap()
            .contains("too long"));
        assert!(form("a cup", "big", "/o")
            .problem()
            .unwrap()
            .contains("metres"));
        assert!(form("a cup", "-1", "/o")
            .problem()
            .unwrap()
            .contains("metres"));
        assert!(form("a cup", "", " ").problem().unwrap().contains("folder"));
        assert!(form("--help", "", "/o").problem().unwrap().contains("dash"));
    }

    fn line(event: &str, extra: Value) -> Value {
        let mut v = json!({"event": event, "batch": "made", "folder": "/o/made/chair-1", "machine": {"card": "RTX 3060"}});
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        v
    }

    #[test]
    fn only_this_makes_lines_are_followed() {
        let out = Path::new("/o/made");
        assert!(is_ours(&line("stage", json!({})), out));
        assert!(is_ours(
            &json!({"event": "not made", "batch": "made", "folder": null}),
            out
        ));
        assert!(!is_ours(
            &json!({"event": "stage", "batch": "other", "folder": "/o/other/x"}),
            out
        ));
        assert!(!is_ours(
            &json!({"event": "stage", "batch": "made", "folder": "/elsewhere/made/x"}),
            out
        ));
    }

    #[test]
    fn the_follower_reads_whole_new_lines_only() {
        let folder = std::env::temp_dir().join(format!("companion-ledger-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("jobs.jsonl");
        std::fs::write(&path, "{\"old\":1}\n").unwrap();
        let mut follower = Follower::from_end(&path);
        assert!(follower.new_lines().is_empty());
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"{\"a\":1}\n{\"b\":").unwrap();
        assert_eq!(follower.new_lines(), [json!({"a": 1})]);
        file.write_all(b"2}\n").unwrap();
        assert_eq!(follower.new_lines(), [json!({"b": 2})]);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_recorded_make_moves_its_bars_forward_only() {
        let expected = [Some(100.0), Some(200.0), None, Some(50.0)];
        let mut job = Job::default();
        let mut rows = Rows::default();
        advance(&mut rows, &job, &expected, Duration::from_secs(50));
        assert_eq!(rows.stage[0].shown(), 0.5);
        advance(&mut rows, &job, &expected, Duration::from_secs(10));
        assert_eq!(rows.stage[0].shown(), 0.5, "never back");
        job.apply(&line(
            "stage",
            json!({"stage": "pictures", "seconds": 90.0}),
        ));
        assert_eq!(job.current(), Some(1));
        assert_eq!(job.card.as_deref(), Some("RTX 3060"));
        advance(&mut rows, &job, &expected, Duration::from_secs(1000));
        assert_eq!(rows.stage[0].shown(), 1.0);
        assert_eq!(rows.stage[1].shown(), progress::CLOCK_BAR_MOST);
        job.apply(&line("stage", json!({"stage": "shape", "seconds": 1.0})));
        advance(&mut rows, &job, &expected, Duration::from_secs(1000));
        assert_eq!(
            rows.stage[2].shown(),
            0.0,
            "no expected time: no invented progress"
        );
        let whole = rows.whole.shown();
        assert!(whole > 0.4 && whole < 0.6, "{whole}");
    }

    #[test]
    fn a_make_that_keeps_earlier_stages_ends_with_every_bar_full() {
        // "Make another" with the same words into the same folder: the pipeline keeps the pictures and the shape,
        // and the ledger has lines for paint and finish only. The whole bar once stopped at 74 %.
        let expected = [Some(100.0), Some(200.0), Some(300.0), Some(50.0)];
        let mut job = Job::default();
        let mut rows = Rows::default();
        advance(&mut rows, &job, &expected, Duration::from_secs(1000));
        job.apply(&line("stage", json!({"stage": "paint", "seconds": 600.0})));
        assert!(job.kept(0) && job.kept(1) && !job.kept(2) && !job.kept(3));
        assert_eq!(
            job.current(),
            Some(3),
            "no stage before a finished one is under way"
        );
        advance(&mut rows, &job, &expected, Duration::from_secs(10));
        assert_eq!(rows.stage[0].shown(), 1.0);
        assert_eq!(rows.stage[1].shown(), 1.0);
        assert_eq!(rows.stage[3].shown(), 0.2);
        job.apply(&line("stage", json!({"stage": "finish", "seconds": 30.0})));
        job.apply(&line(
            "job",
            json!({"folder": "/tmp/out/a-long-wooden-spear", "slug": "a-long-wooden-spear", "passed": true}),
        ));
        assert_eq!(job.current(), None);
        advance(&mut rows, &job, &expected, Duration::from_secs(0));
        assert_eq!(rows.whole.shown(), 1.0);
        assert!(rows.stage.iter().all(|bar| bar.shown() == 1.0));
    }

    #[test]
    fn a_made_job_with_no_stage_lines_is_kept_whole() {
        let mut job = Job::default();
        job.apply(&line(
            "job",
            json!({"folder": "/tmp/out/a-chair", "slug": "a-chair", "passed": true}),
        ));
        let mut rows = Rows::default();
        advance(&mut rows, &job, &[None; 4], Duration::from_secs(0));
        assert!((0..STAGES.len()).all(|i| job.kept(i)));
        assert_eq!(rows.whole.shown(), 1.0);
    }

    #[test]
    fn expected_times_come_from_this_cards_past_stages() {
        let ledger = [
            line("stage", json!({"stage": "pictures", "seconds": 10.0})),
            line("stage", json!({"stage": "pictures", "seconds": 30.0})),
            json!({"event": "stage", "stage": "pictures", "seconds": 999.0, "machine": {"card": "other"}}),
            line("load", json!({"stage": "shape", "seconds": 5.0})),
        ]
        .map(|v| v.to_string())
        .join("\n");
        assert_eq!(
            expected_seconds(&ledger, Some("RTX 3060")),
            [Some(20.0), None, None, None]
        );
        assert_eq!(expected_seconds(&ledger, None)[0], Some(30.0));
    }

    #[test]
    fn the_ends_are_said_in_words() {
        let mut job = Job::default();
        job.apply(&line(
            "not made",
            json!({"why": "a weapon of war is not made"}),
        ));
        assert_eq!(
            end_words(job.end.as_ref().unwrap()),
            "Not made: a weapon of war is not made"
        );
        job.apply(&line(
            "aborted",
            json!({"stage": "paint", "error": "out of card memory"}),
        ));
        assert_eq!(
            end_words(job.end.as_ref().unwrap()),
            "Stopped in paint: out of card memory"
        );
        let made = End::Made {
            folder: PathBuf::from("/no/such/folder"),
            slug: "chair".into(),
            passed: false,
            failed: vec!["stands".into()],
        };
        assert_eq!(end_words(&made), "Made, with 1 failed check: stands");
    }

    #[test]
    fn the_result_line_counts_the_sidecars_checks() {
        let passed = |name: &str, ok: bool| json!({"name": name, "passed": ok});
        let sidecar = json!({"checks": {
            "picture": [passed("one thing", true), {"name": "lettering", "passed": false, "counted": false}],
            "shape": [passed("stands", true), passed("whole", true)],
            "paint": [passed("colours", true)],
        }});
        assert_eq!(
            result_words(&sidecar).unwrap(),
            "Made: passes 4 of 4 checks"
        );
        let sidecar = json!({"checks": {"shape": [passed("stands", false), passed("whole", false), passed("x", true)]}});
        assert_eq!(
            result_words(&sidecar).unwrap(),
            "Made, with 2 failed checks of 3: stands; whole"
        );
        assert_eq!(result_words(&json!({})), None);
        let folder = std::env::temp_dir().join(format!("companion-sidecar-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("chair.provenance.json"), sidecar.to_string()).unwrap();
        let made = End::Made {
            folder: folder.clone(),
            slug: "chair".into(),
            passed: false,
            failed: vec![],
        };
        assert_eq!(
            end_words(&made),
            "Made, with 2 failed checks of 3: stands; whole"
        );
        std::fs::remove_dir_all(&folder).unwrap();
    }
}

/// A whole make through a stand-in launcher (a shell script that writes the ledger and the sidecar as the pipeline
/// does), and a cancelled one: the window's own path from "Make" to its last line.
#[cfg(all(test, unix))]
mod through_a_stand_in {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    const STAND_IN: &str = r#"#!/bin/sh
out="$4"
ledger="$REPO/metrics/jobs.jsonl"
batch=$(basename "$out")
job="$out/chair-1"
mkdir -p "$job"
[ "$2" = "slow" ] && exec sleep 60
if [ "$2" = "dies" ]; then
  printf '{"event":"stage","stage":"pictures","seconds":2.5,"batch":"%s","folder":"%s","machine":{"card":"T"}}\n' "$batch" "$job" >> "$ledger"
  echo "MemoryError: out of memory"
  kill -9 $$
fi
for stage in pictures shape paint finish; do
  printf '{"event":"stage","stage":"%s","seconds":2.5,"batch":"%s","folder":"%s","machine":{"card":"T"}}\n' "$stage" "$batch" "$job" >> "$ledger"
done
printf '{"checks":{"shape":[{"name":"stands","passed":true},{"name":"whole","passed":false}]}}' > "$job/chair.provenance.json"
printf '{"event":"job","batch":"%s","folder":"%s","slug":"chair","passed":false,"failed":["whole"]}\n' "$batch" "$job" >> "$ledger"
echo "made $job"
"#;

    fn stand_in(name: &str) -> (PathBuf, PathBuf) {
        let root =
            std::env::temp_dir().join(format!("companion-make-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("repo/metrics")).unwrap();
        let script = root.join("model-maker");
        let text = STAND_IN.replace("$REPO", &root.join("repo").display().to_string());
        std::fs::write(&script, text).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        (root, script)
    }

    fn until_ended(making: &mut Making, most: Duration) {
        let since = Instant::now();
        while making.poll() && since.elapsed() < most {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn a_make_ends_with_the_sidecars_result_and_its_folder() {
        let (root, launcher) = stand_in("whole");
        let form = Form {
            prompt: "an old looking chair".into(),
            out: root.join("made").display().to_string(),
            ..Form::default()
        };
        let mut making = Making::start(&launcher, &root.join("repo"), &form).unwrap();
        until_ended(&mut making, Duration::from_secs(10));
        assert_eq!(
            making.outcome_words().unwrap(),
            "Made, with 1 failed check of 2: whole"
        );
        assert!(making.folder().unwrap().ends_with("made/chair-1"));
        assert_eq!(making.job.done, [Some(2.5); 4]);
        assert_eq!(making.rows.whole.shown(), 1.0);
        assert_eq!(making.left_running, Some(false));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The Windows card test's seventh try (28 Nov 2026): the make's Python ran the test PC out of memory and ended. The
    /// window must outlive a child that dies hard and say so, with its last words in the Details.
    #[test]
    fn a_child_killed_hard_ends_without_a_result_and_the_window_goes_on() {
        let (root, launcher) = stand_in("dies");
        let form = Form {
            prompt: "dies".into(),
            out: root.join("made").display().to_string(),
            ..Form::default()
        };
        let mut making = Making::start(&launcher, &root.join("repo"), &form).unwrap();
        until_ended(&mut making, Duration::from_secs(10));
        assert_eq!(
            making.outcome_words().unwrap(),
            "The make ended without a result (exit code not known); the Details show why."
        );
        assert!(making
            .running
            .lines()
            .iter()
            .any(|l| l == "MemoryError: out of memory"));
        assert_eq!(making.job.done[0], Some(2.5));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cancel_leaves_nothing_running_and_keeps_the_folder() {
        let (root, launcher) = stand_in("cancel");
        let form = Form {
            prompt: "slow".into(),
            out: root.join("made").display().to_string(),
            ..Form::default()
        };
        let mut making = Making::start(&launcher, &root.join("repo"), &form).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert!(making.poll());
        making.running.stop();
        until_ended(&mut making, Duration::from_secs(20));
        assert_eq!(
            making.outcome_words().unwrap(),
            "Cancelled: nothing of it is left running. Its folder is kept as it is."
        );
        assert!(root.join("made/chair-1").is_dir());
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// Cancel on Windows through the real built `model-maker.exe` (no card): the launcher's job object must end Python and
/// a detached grandchild that no console event reaches. The launcher is COMPANION_TEST_LAUNCHER, else the workspace's
/// release or debug build; the Python is COMPANION_TEST_PYTHON, else `python` on the PATH. Either missing fails the
/// test: it is never skipped on Windows.
#[cfg(all(test, windows))]
mod cancel_on_windows {
    use super::*;

    const STAND_IN_MAIN: &str = r#"
import json, os, subprocess, sys, time
out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(os.path.join(out, "chair-1"), exist_ok=True)
# DETACHED_PROCESS: no console, so only the launcher's job object can end it.
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], creationflags=0x8)
with open(os.path.join(out, "pids.json"), "w") as file:
    json.dump({"interpreter": os.getpid(), "grandchild": grandchild.pid}, file)
time.sleep(600)
"#;

    fn launcher() -> PathBuf {
        if let Some(named) = std::env::var_os("COMPANION_TEST_LAUNCHER") {
            return PathBuf::from(named);
        }
        let target = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
        ["release", "debug"]
            .iter()
            .map(|profile| target.join(profile).join("model-maker.exe"))
            .find(|p| p.is_file())
            .expect("no built model-maker.exe: build the launcher, or name it in COMPANION_TEST_LAUNCHER")
    }

    fn python() -> PathBuf {
        if let Some(named) = std::env::var_os("COMPANION_TEST_PYTHON") {
            return PathBuf::from(named);
        }
        let found = std::process::Command::new("where")
            .arg("python")
            .output()
            .expect("`where python` could not run");
        let text = String::from_utf8_lossy(&found.stdout);
        PathBuf::from(
            text.lines()
                .next()
                .expect("no python on the PATH: name one in COMPANION_TEST_PYTHON")
                .trim(),
        )
    }

    fn alive(pid: u32) -> bool {
        let listed = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .output()
            .expect("tasklist could not run");
        String::from_utf8_lossy(&listed.stdout).contains(&format!("\"{pid}\""))
    }

    fn kill(pid: u32) {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .output();
    }

    #[test]
    fn cancel_ends_the_launcher_python_and_a_detached_grandchild_and_keeps_the_folder() {
        let root = std::env::temp_dir().join(format!("companion-cancel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repository = root.join("repository");
        let package = repository.join("python").join("model_maker");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(package.join("__init__.py"), "").unwrap();
        std::fs::write(package.join("__main__.py"), STAND_IN_MAIN).unwrap();
        let launcher_copy = repository.join("model-maker.exe");
        std::fs::copy(launcher(), &launcher_copy).unwrap();
        // The launcher runs MODEL_MAKER_PYTHON when it is set (as the bash command); only this test sets it.
        std::env::set_var("MODEL_MAKER_PYTHON", python());

        let form = Form {
            prompt: "slow".into(),
            out: root.join("made").display().to_string(),
            ..Form::default()
        };
        let mut making = Making::start(&launcher_copy, &repository, &form).unwrap();
        let pids_file = making.out.join("pids.json");
        let since = Instant::now();
        while !pids_file.is_file() && since.elapsed() < Duration::from_secs(30) {
            assert!(
                making.poll(),
                "the make ended early: {:?}",
                making.running.lines()
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        std::thread::sleep(Duration::from_millis(300));
        let pids: Value =
            serde_json::from_str(&std::fs::read_to_string(&pids_file).unwrap()).unwrap();
        let three = [
            ("launcher", making.running.id()),
            ("python", pids["interpreter"].as_u64().unwrap() as u32),
            ("grandchild", pids["grandchild"].as_u64().unwrap() as u32),
        ];
        assert!(three.iter().all(|(_, pid)| alive(*pid)), "{three:?}");

        making.running.stop();
        let since = Instant::now();
        while making.poll() && since.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(100));
        }
        let mut left: Vec<(&str, u32)> = three.to_vec();
        while !left.is_empty() && since.elapsed() < Duration::from_secs(20) {
            left.retain(|(_, pid)| alive(*pid));
            std::thread::sleep(Duration::from_millis(250));
        }
        for (_, pid) in &left {
            kill(*pid);
        }
        assert!(left.is_empty(), "still running 20 s after Cancel: {left:?}");
        assert_eq!(
            making.outcome_words().as_deref(),
            Some("Cancelled. Its folder is kept as it is.")
        );
        assert!(root.join("made").join("chair-1").is_dir());
        drop(making);
        let _ = std::fs::remove_dir_all(&root);
    }
}
