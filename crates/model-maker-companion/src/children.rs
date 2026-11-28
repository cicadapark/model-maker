//! The programs the window starts: the launcher itself (`model-maker setup`, `model-maker make ...`), with no console
//! window of its own, its output kept for the Details fold as it is (never read for meaning), and a way to stop it
//! with everything it started (plan section 3, "Children").
//!
//! Unix: the launcher becomes Python (exec) in a process group of its own; stopping sends SIGTERM to the group, waits
//! 10 s, then SIGKILL (as the pipeline's processes.py). Windows: the launcher put itself into a job object that ends
//! everything in it when the launcher ends, so ending the launcher ends the make.

use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
#[cfg(unix)]
use std::time::Duration;
use std::time::Instant;

use model_maker_launcher::platform;

/// The Details fold keeps the last lines only.
pub const KEPT_LINES: usize = 400;
/// SIGTERM first; SIGKILL after this (processes.py's own wait).
#[cfg(unix)]
pub const POLITE_WAIT: Duration = Duration::from_secs(10);

/// The launcher beside this program (the archive's layout), else none.
pub fn launcher_beside(this_program: &Path) -> Option<PathBuf> {
    let (system, _) = platform::here();
    let launcher = this_program
        .parent()?
        .join(platform::program(system, "model-maker"));
    launcher.is_file().then_some(launcher)
}

pub struct Running {
    process: std::process::Child,
    lines: Arc<Mutex<Vec<String>>>,
    /// The threads reading its output: its end is given once they have read all of it (or OUTPUT_WAIT after it ended).
    readers: Vec<std::thread::JoinHandle<()>>,
    /// When the process itself was first seen ended.
    exited: Option<Instant>,
    pub started: Instant,
    stop_asked: Option<Instant>,
}

/// After the program ends, how long its end waits for its last output to be read: a program it started that still
/// holds the output open must not hold the end back for ever.
const OUTPUT_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ended {
    pub code: Option<i32>,
    /// It was asked to stop (Cancel), so its end is not a failure of its own.
    pub stopped_by_us: bool,
}

fn kept(lines: &Arc<Mutex<Vec<String>>>, line: String) {
    if let Ok(mut all) = lines.lock() {
        all.push(line);
        let extra = all.len().saturating_sub(KEPT_LINES);
        all.drain(..extra);
    }
}

/// The longest line kept for the Details fold; the rest of a longer one is left out.
pub const LINE_MOST: usize = 300;

/// Splits a program's output into lines at `\n` and at `\r` (a progress line that rewrites itself), each at most
/// `LINE_MOST` bytes, control characters left out.
#[derive(Default)]
pub struct Splitter {
    partial: Vec<u8>,
    cut: bool,
}

impl Splitter {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        for &byte in bytes {
            if byte == b'\n' || byte == b'\r' {
                if !self.partial.is_empty() || self.cut {
                    lines.push(self.take());
                }
            } else if self.partial.len() < LINE_MOST {
                self.partial.push(byte);
            } else {
                self.cut = true;
            }
        }
        lines
    }

    /// What is left when the output ends.
    pub fn rest(&mut self) -> Option<String> {
        (!self.partial.is_empty()).then(|| self.take())
    }

    fn take(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.partial);
        let mut line: String = text
            .chars()
            .map(|c| if c == '\t' { ' ' } else { c })
            .filter(|c| !c.is_control())
            .collect();
        if self.cut {
            line.push('…');
        }
        self.partial.clear();
        self.cut = false;
        line
    }
}

fn read_lines(
    mut from: impl Read + Send + 'static,
    lines: Arc<Mutex<Vec<String>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut splitter = Splitter::default();
        let mut chunk = [0u8; 4096];
        while let Ok(n) = from.read(&mut chunk) {
            if n == 0 {
                break;
            }
            for line in splitter.push(&chunk[..n]) {
                kept(&lines, line);
            }
        }
        if let Some(line) = splitter.rest() {
            kept(&lines, line);
        }
    })
}

/// Linux refuses to start a program while any process holds it open for writing ("Text file busy"), which a
/// program just written can be for a moment (another thread's child between its fork and its exec): tried again.
const BUSY_TRIES: u32 = 20;

fn spawned(command: &mut Command) -> std::io::Result<std::process::Child> {
    let mut tries = 1;
    loop {
        match command.spawn() {
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && tries < BUSY_TRIES => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            other => return other,
        }
    }
}

/// Starts `program` with `words`, its own environment plus `settings`.
pub fn start(
    program: &Path,
    words: &[OsString],
    settings: &[(OsString, OsString)],
) -> Result<Running, String> {
    let mut command = Command::new(program);
    command
        .args(words)
        .envs(settings.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    platform::quiet(&mut command);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut process = spawned(&mut command)
        .map_err(|e| format!("{} could not be started ({e})", program.display()))?;
    let lines = Arc::new(Mutex::new(Vec::new()));
    let mut readers = Vec::new();
    if let Some(out) = process.stdout.take() {
        readers.push(read_lines(out, lines.clone()));
    }
    if let Some(err) = process.stderr.take() {
        readers.push(read_lines(err, lines.clone()));
    }
    Ok(Running {
        process,
        lines,
        readers,
        exited: None,
        started: Instant::now(),
        stop_asked: None,
    })
}

#[cfg(unix)]
fn signal_group(group: u32, signal: &str) -> bool {
    // By its full path: never a `kill` found first in PATH.
    Command::new("/bin/kill")
        .args([signal, "--", &format!("-{group}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

impl Running {
    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// Its end, once it has ended (never blocks). After a stop on Unix, SIGKILL follows SIGTERM after 10 s.
    pub fn ended(&mut self) -> Option<Ended> {
        #[cfg(unix)]
        if let Some(asked) = self.stop_asked {
            if asked.elapsed() >= POLITE_WAIT && self.anything_left() {
                signal_group(self.process.id(), "-KILL");
            }
        }
        let status = self.process.try_wait().ok()??;
        #[cfg(unix)]
        if self.stop_asked.is_some() && self.anything_left() {
            // The launcher (Python) has ended but something it started has not: wait for the group too.
            return None;
        }
        // Its last lines belong in the Details when its end is shown.
        let exited = *self.exited.get_or_insert_with(Instant::now);
        if !self.readers.iter().all(|r| r.is_finished()) && exited.elapsed() < OUTPUT_WAIT {
            return None;
        }
        Some(Ended {
            code: status.code(),
            stopped_by_us: self.stop_asked.is_some(),
        })
    }

    /// Asks it to stop: Unix, SIGTERM to its whole group (SIGKILL follows in `ended`); Windows, the launcher is ended
    /// and its job object ends everything it started.
    pub fn stop(&mut self) {
        if self.stop_asked.is_some() {
            return;
        }
        self.stop_asked = Some(Instant::now());
        #[cfg(unix)]
        signal_group(self.process.id(), "-TERM");
        #[cfg(not(unix))]
        let _ = self.process.kill();
    }

    pub fn stopping(&self) -> bool {
        self.stop_asked.is_some()
    }

    /// The launcher's process id (the Windows Cancel test counts it).
    #[cfg(all(test, windows))]
    pub fn id(&self) -> u32 {
        self.process.id()
    }

    /// Whether any process of its group is still there (Unix); elsewhere, whether the launcher itself is.
    pub fn anything_left(&mut self) -> bool {
        #[cfg(unix)]
        {
            let _ = self.process.try_wait();
            signal_group(self.process.id(), "-0")
        }
        #[cfg(not(unix))]
        {
            matches!(self.process.try_wait(), Ok(None))
        }
    }
}

impl Drop for Running {
    /// Closing the window ends what it started: nothing is left running, the group's other members too when its
    /// first program has already ended (closed during Cancel's wait).
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.anything_left() {
            signal_group(self.process.id(), "-KILL");
        }
        if matches!(self.process.try_wait(), Ok(None)) {
            let _ = self.process.kill();
        }
        let _ = self.process.wait();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn wait_for_end(running: &mut Running, most: Duration) -> Option<Ended> {
        let since = Instant::now();
        while since.elapsed() < most {
            if let Some(ended) = running.ended() {
                return Some(ended);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        None
    }

    #[test]
    fn output_is_kept_line_by_line_and_the_exit_code_given() {
        let words: Vec<OsString> = ["-c", "echo one; echo two >&2; exit 3"]
            .iter()
            .map(OsString::from)
            .collect();
        let mut running = start(Path::new("/bin/sh"), &words, &[]).unwrap();
        let ended = wait_for_end(&mut running, Duration::from_secs(10)).unwrap();
        assert_eq!(
            ended,
            Ended {
                code: Some(3),
                stopped_by_us: false
            }
        );
        std::thread::sleep(Duration::from_millis(100));
        let mut lines = running.lines();
        lines.sort();
        assert_eq!(lines, ["one", "two"]);
    }

    #[test]
    fn stop_ends_the_child_and_its_grandchildren() {
        // A grandchild that ignores nothing, and a child that waits on it: both must be gone after stop.
        let words: Vec<OsString> = ["-c", "sleep 60 & sleep 60; wait"]
            .iter()
            .map(OsString::from)
            .collect();
        let mut running = start(Path::new("/bin/sh"), &words, &[]).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert!(running.anything_left());
        running.stop();
        let ended = wait_for_end(&mut running, Duration::from_secs(15)).unwrap();
        assert!(ended.stopped_by_us);
        assert!(!running.anything_left(), "no process of the group is left");
    }

    #[test]
    fn a_child_that_ignores_sigterm_is_killed_after_the_wait() {
        let words: Vec<OsString> = ["-c", "trap '' TERM; sleep 30 & wait; sleep 30"]
            .iter()
            .map(OsString::from)
            .collect();
        let mut running = start(Path::new("/bin/sh"), &words, &[]).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        running.stop();
        let ended = wait_for_end(&mut running, POLITE_WAIT + Duration::from_secs(10)).unwrap();
        assert!(ended.stopped_by_us);
        assert!(!running.anything_left());
    }

    #[test]
    fn the_settings_reach_the_child_and_a_missing_program_is_said() {
        let words: Vec<OsString> = ["-c", "echo \"$COMPANION_TEST\""]
            .iter()
            .map(OsString::from)
            .collect();
        let settings = [(OsString::from("COMPANION_TEST"), OsString::from("seen"))];
        let mut running = start(Path::new("/bin/sh"), &words, &settings).unwrap();
        wait_for_end(&mut running, Duration::from_secs(10)).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(running.lines(), ["seen"]);
        let refused = start(Path::new("/no/such/program"), &[], &[])
            .err()
            .unwrap();
        assert!(refused.contains("could not be started"));
    }
}

#[cfg(test)]
mod splitter_tests {
    use super::*;

    #[test]
    fn lines_break_at_carriage_returns_are_capped_and_cleaned() {
        let mut splitter = Splitter::default();
        assert_eq!(splitter.push(b"10%\r20%\r"), ["10%", "20%"]);
        assert_eq!(splitter.push(b"a\tb\x1b[0mc\r\n"), ["a b[0mc"]);
        assert!(splitter.push(b"half").is_empty());
        assert_eq!(splitter.push(b" line\n"), ["half line"]);
        let long = vec![b'x'; LINE_MOST * 3];
        assert!(splitter.push(&long).is_empty());
        let cut = splitter.push(b"\n");
        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].chars().count(), LINE_MOST + 1);
        assert!(cut[0].ends_with('…'));
        assert!(splitter.push(b"end").is_empty());
        assert_eq!(splitter.rest().as_deref(), Some("end"));
        assert_eq!(splitter.rest(), None);
    }
}
