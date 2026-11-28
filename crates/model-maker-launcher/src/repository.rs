//! Where the repository is (one rule: from the program's own real path) and which version of it this is.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The program sits in the repository's top folder (the Windows and macOS archives), in its `bin/` (the Linux
/// archive), or in `target/<profile>/` or `target/<triple>/<profile>/` when built here. Only those three places are
/// looked at: a copy of the program in a shared folder must never run a `python/model_maker` planted above it.
const PACKAGE_MARK: &str = "python/model_maker/__init__.py";

#[derive(Debug, PartialEq, Eq)]
pub struct NotBeside(pub PathBuf);

impl std::fmt::Display for NotBeside {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: model-maker's Python side (python/model_maker) is not beside this program; run the model-maker that \
             came in the repository's folder",
            self.0.display()
        )
    }
}

/// The folders the repository may be, for a program in `folder`: the folder itself; its parent when it is `bin`;
/// the folder above `target` when it is `target/<profile>` or `target/<triple>/<profile>`.
fn allowed_places(folder: &Path) -> Vec<&Path> {
    let named = |path: &Path, name: &str| path.file_name().is_some_and(|n| n == name);
    let mut places = vec![folder];
    if named(folder, "bin") {
        places.extend(folder.parent());
    }
    for target in folder.ancestors().skip(1).take(2) {
        if named(target, "target") {
            places.extend(target.parent());
        }
    }
    places
}

/// The repository for a program at `program` (its real path, links resolved by the caller).
pub fn beside(program: &Path) -> Result<PathBuf, NotBeside> {
    let folder = program.parent().unwrap_or(program);
    allowed_places(folder)
        .into_iter()
        .find(|candidate| candidate.join(PACKAGE_MARK).is_file())
        .map(Path::to_path_buf)
        .ok_or_else(|| NotBeside(folder.to_path_buf()))
}

/// The repository of the running program.
pub fn of_this_program() -> Result<PathBuf, NotBeside> {
    let program = std::env::current_exe().map_err(|_| NotBeside(PathBuf::from("model-maker")))?;
    beside(&crate::paths::real(&program))
}

/// The repository's version, by the Python side's own rule (pipeline.repository_version): `git describe` in a
/// checkout; in a release made by `git archive`, the commit written into `version.txt`; else "unknown".
pub fn version(repository: &Path) -> String {
    let package = repository.join("python").join("model_maker");
    let described = crate::platform::quiet(
        Command::new("git")
            .arg("-C")
            .arg(&package)
            .args(["describe", "--always", "--dirty"]),
    )
    .output()
    .ok()
    .filter(|done| done.status.success())
    .map(|done| String::from_utf8_lossy(&done.stdout).trim().to_string());
    if let Some(described) = described.filter(|d| !d.is_empty()) {
        return described;
    }
    written_version(&package.join("version.txt"))
}

fn written_version(file: &Path) -> String {
    match std::fs::read_to_string(file) {
        Ok(text) if !text.trim().is_empty() && !text.contains("$Format") => text.trim().to_string(),
        _ => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary(name: &str) -> PathBuf {
        let folder =
            std::env::temp_dir().join(format!("mm-launcher-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    fn fake_repository(root: &Path) {
        std::fs::create_dir_all(root.join("python/model_maker")).unwrap();
        std::fs::write(root.join(PACKAGE_MARK), "").unwrap();
    }

    #[test]
    fn found_from_the_top_folder_bin_and_a_build_folder() {
        let root = temporary("found");
        fake_repository(&root);
        for program in [
            "model-maker.exe",
            "bin/model-maker",
            "target/release/model-maker",
            "target/x86_64-pc-windows-msvc/release/model-maker.exe",
        ] {
            assert_eq!(beside(&root.join(program)).unwrap(), root, "{program}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn refused_when_the_python_side_is_not_beside_it() {
        let root = temporary("refused");
        fake_repository(&root);
        let too_deep = root.join("a/b/c/d/model-maker");
        let refused = beside(&too_deep).unwrap_err();
        assert!(refused.to_string().contains("python/model_maker"));
        let elsewhere = temporary("elsewhere");
        assert!(beside(&elsewhere.join("model-maker")).is_err());
        // A copy in a shared folder, a `python/model_maker` planted one level above it.
        assert!(beside(&root.join("shared/model-maker")).is_err());
        assert!(beside(&root.join("target/a/b/c/model-maker")).is_err());
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::remove_dir_all(&elsewhere).unwrap();
    }

    #[test]
    fn found_whatever_the_working_folder_is() {
        let root = temporary("cwd");
        fake_repository(&root);
        let other = temporary("cwd-other");
        let before = std::env::current_dir().unwrap();
        std::env::set_current_dir(&other).unwrap();
        let found = beside(&root.join("bin/model-maker"));
        std::env::set_current_dir(before).unwrap();
        assert_eq!(found.unwrap(), root);
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::remove_dir_all(&other).unwrap();
    }

    #[test]
    fn written_version_as_python_reads_it() {
        let root = temporary("version");
        let file = root.join("version.txt");
        std::fs::write(&file, "$Format:%h$\n").unwrap();
        assert_eq!(written_version(&file), "unknown");
        std::fs::write(&file, "64cfae1\n").unwrap();
        assert_eq!(written_version(&file), "64cfae1");
        std::fs::remove_file(&file).unwrap();
        assert_eq!(written_version(&file), "unknown");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn this_repository_has_a_version() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        assert!(!version(&repository).is_empty());
    }
}
