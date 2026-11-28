//! The built program, run as a person runs it: every word that is not the launcher's own goes to Python unchanged;
//! on Linux and macOS the launcher becomes Python (the same process id). The Windows ends (hard kill, Ctrl+C, the
//! console closed, a grandchild) are tested on Windows by tools/ci/launcher_ends.py.
//!
//! Every test here starts programs that were written by a test. On Linux a program cannot start while any process
//! holds it open for writing ("Text file busy"), and a child forked by another test's thread at that moment holds
//! a copy of the writer's handle until it starts its own program. So the stand-ins are written once, before any
//! test starts a program, and the tests run one at a time.

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard, OnceLock};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Held for a whole test: nothing else in this program writes a file or starts a program meanwhile.
fn one_at_a_time() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A stand-in repository with the built program in its `bin/`, and two stand-in Pythons.
struct StandIns {
    root: PathBuf,
    /// Prints its process id, its words (one a line) and two variables; ends with 5.
    python: PathBuf,
    /// Prints its whole environment (for the Linux-only comparison with the bash command).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    environment_printer: PathBuf,
}

fn executable(path: &Path, script: &str) {
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Made once, by the first test (which holds `one_at_a_time`), so no file is written while a program starts.
fn stand_ins() -> &'static StandIns {
    static MADE: OnceLock<StandIns> = OnceLock::new();
    MADE.get_or_init(|| {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("passthrough");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("python/model_maker")).unwrap();
        std::fs::write(root.join("python/model_maker/__init__.py"), "").unwrap();
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::copy(
            env!("CARGO_BIN_EXE_model-maker"),
            root.join("bin/model-maker"),
        )
        .unwrap();
        let python = root.join("python-stand-in");
        executable(
            &python,
            "#!/bin/sh\necho \"pid $$\"\nfor word in \"$@\"; do echo \"word $word\"; done\n\
             echo \"PYTHONPATH $PYTHONPATH\"\necho \"HF_HUB_OFFLINE $HF_HUB_OFFLINE\"\nexit 5\n",
        );
        let environment_printer = root.join("environment-printer");
        executable(&environment_printer, "#!/bin/sh\nenv\n");
        StandIns {
            root,
            python,
            environment_printer,
        }
    })
}

#[test]
fn words_reach_python_unchanged_in_the_same_process() {
    let _alone = one_at_a_time();
    let stand_ins = stand_ins();
    let launcher = stand_ins.root.join("bin/model-maker");
    let child = Command::new(&launcher)
        .args([
            "make",
            "an old looking chair",
            "--out",
            "a dir/with \"quotes\"",
            "--size",
            "0.9",
        ])
        .env("MODEL_MAKER_PYTHON", &stand_ins.python)
        .env_remove("PYTHONPATH")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let output = child.wait_with_output().unwrap();
    let said = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(5),
        "Python's own exit code: {said}"
    );
    assert!(
        said.contains(&format!("pid {pid}\n")),
        "the launcher became Python: {said}"
    );
    let words: Vec<&str> = said
        .lines()
        .filter_map(|l| l.strip_prefix("word "))
        .collect();
    assert_eq!(
        words,
        [
            "-m",
            "model_maker",
            "make",
            "an old looking chair",
            "--out",
            "a dir/with \"quotes\"",
            "--size",
            "0.9"
        ]
    );
    let canonical = stand_ins.root.canonicalize().unwrap();
    assert!(
        said.contains(&format!(
            "PYTHONPATH {}\n",
            canonical.join("python").display()
        )),
        "{said}"
    );
    assert!(said.contains("HF_HUB_OFFLINE 1\n"));
}

#[test]
fn the_launchers_own_words_are_not_passed_on() {
    let _alone = one_at_a_time();
    let stand_ins = stand_ins();
    let output = Command::new(stand_ins.root.join("bin/model-maker"))
        .arg("--version")
        .env("MODEL_MAKER_PYTHON", &stand_ins.python)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&output.stdout);
    assert!(said.starts_with("model-maker "), "{said}");
    assert!(!said.contains("pid "), "Python was not started: {said}");
}

#[test]
fn no_environment_is_said_plainly() {
    let _alone = one_at_a_time();
    let stand_ins = stand_ins();
    let output = Command::new(stand_ins.root.join("bin/model-maker"))
        .arg("words")
        .env_remove("MODEL_MAKER_PYTHON")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("model-maker setup"));
}

/// The bash `model-maker`, run with a stand-in Python that prints its environment, sets the same values as the
/// launcher's `environment::for_python`.
#[cfg(target_os = "linux")]
#[test]
fn equal_to_the_bash_command() {
    use model_maker_launcher::environment::for_python;
    use model_maker_launcher::platform::System;
    use std::ffi::OsString;

    let _alone = one_at_a_time();
    let stand_ins = stand_ins();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let output = Command::new(repository.join("model-maker"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("MODEL_MAKER_PYTHON", &stand_ins.environment_printer)
        .output()
        .unwrap();
    let printed = String::from_utf8_lossy(&output.stdout);
    let bash: Vec<(&str, &str)> = printed.lines().filter_map(|l| l.split_once('=')).collect();
    let ours = for_python(
        System::Linux,
        &repository,
        &repository.join("models"),
        &repository.join("sources"),
        [(OsString::from("PATH"), OsString::from("/usr/bin:/bin"))],
    );
    assert!(!ours.set.is_empty());
    for (name, value) in &ours.set {
        let name = name.to_str().unwrap();
        let theirs = bash.iter().find(|(n, _)| *n == name).map(|(_, v)| *v);
        assert_eq!(theirs, value.to_str(), "{name}");
    }
}
