//! The built program writes setup's progress file (MODEL_MAKER_PROGRESS) as the companion reads it: here a setup
//! refused before it changes anything ends with one `stopped` line saying why. The program runs from a stand-in
//! repository (its bin/), so the test does not depend on where cargo put it.

use std::process::Command;

#[test]
fn a_refused_setup_writes_one_stopped_line_with_its_why() {
    let folder = std::env::temp_dir().join(format!("mm-launcher-progress-{}", std::process::id()));
    let repository = folder.join("repository");
    std::fs::create_dir_all(repository.join("python/model_maker")).unwrap();
    std::fs::create_dir_all(repository.join("bin")).unwrap();
    std::fs::write(repository.join("python/model_maker/__init__.py"), "").unwrap();
    let program = repository.join("bin").join(
        std::path::Path::new(env!("CARGO_BIN_EXE_model-maker"))
            .file_name()
            .unwrap(),
    );
    std::fs::copy(env!("CARGO_BIN_EXE_model-maker"), &program).unwrap();
    let file = folder.join("progress.jsonl");
    let done = Command::new(&program)
        .args(["setup", "--lock", "no-such-lock"])
        .env("MODEL_MAKER_PROGRESS", &file)
        .output()
        .unwrap();
    assert!(!done.status.success());
    let written = std::fs::read_to_string(&file).unwrap_or_else(|e| {
        panic!(
            "no progress file ({e}): {}",
            String::from_utf8_lossy(&done.stderr)
        )
    });
    let lines: Vec<&str> = written.lines().collect();
    assert_eq!(lines.len(), 1, "{written}");
    assert!(
        lines[0].starts_with("{\"schema\":1,\"event\":\"stopped\",\"why\":\"")
            && lines[0].contains("no-such-lock"),
        "{}",
        lines[0]
    );
    assert!(written.ends_with("}\n"));
    std::fs::remove_dir_all(&folder).unwrap();
}
