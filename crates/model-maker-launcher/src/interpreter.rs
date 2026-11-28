//! The CPython uv installs: checked against the publisher's own sums, and its one copyleft part removed.

use std::path::{Path, PathBuf};

use crate::download::sha256_of;
use crate::platform::{System, Target};

/// Modules of the standalone CPython builds that carry a copyleft library: `_dbm` links Berkeley DB (the
/// Sleepycat licence) in the Linux build; `_gdbm` would be GNU dbm (GPL; no build has it today). Nothing in
/// model-maker or its locks imports `dbm` (shown in docs/licences.md), and Python's `dbm` falls back to `dbm.dumb`
/// without them. Removed on every system, so every PC's interpreter is the same; the licence check refuses them.
pub const REFUSED_MODULES: &[&str] = &["_dbm", "_gdbm"];

/// The interpreter's folder under `.tools/python`.
pub fn home(tools: &Path, target: &Target) -> PathBuf {
    tools.join("python").join(target.interpreter_folder)
}

/// Where the interpreter's compiled standard modules are.
pub fn compiled_modules(home: &Path, system: System) -> PathBuf {
    match system {
        System::Windows => home.join("DLLs"),
        _ => home.join("lib").join("python3.11").join("lib-dynload"),
    }
}

/// Files of the interpreter whose sha256 is not the publisher's (or that are missing).
pub fn unequal_files(home: &Path, target: &Target) -> Vec<String> {
    target
        .interpreter_sums
        .iter()
        .filter(|f| sha256_of(&home.join(f.file)).ok().as_deref() != Some(f.sha256))
        .map(|f| f.file.to_string())
        .collect()
}

fn is_refused_module(file_name: &str) -> bool {
    REFUSED_MODULES.iter().any(|module| {
        file_name
            .strip_prefix(module)
            .is_some_and(|rest| rest.starts_with('.'))
    })
}

/// The refused modules removed from the interpreter's compiled modules; their names.
pub fn refused_modules_removed(home: &Path, system: System) -> std::io::Result<Vec<String>> {
    let folder = compiled_modules(home, system);
    let Ok(entries) = std::fs::read_dir(&folder) else {
        return Ok(Vec::new());
    };
    let mut removed = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_refused_module(&name) {
            std::fs::remove_file(entry.path())?;
            removed.push(name);
        }
    }
    removed.sort();
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{target, Processor};

    #[test]
    fn only_the_refused_modules_are_named() {
        assert!(is_refused_module("_dbm.cpython-311-x86_64-linux-gnu.so"));
        assert!(is_refused_module("_dbm.cpython-311-darwin.so"));
        assert!(is_refused_module("_gdbm.cpython-311-x86_64-linux-gnu.so"));
        assert!(!is_refused_module("_dbmx.so"));
        assert!(!is_refused_module(
            "_decimal.cpython-311-x86_64-linux-gnu.so"
        ));
    }

    #[test]
    fn removal_takes_only_dbm_from_lib_dynload() {
        let home = std::env::temp_dir().join(format!("mm-launcher-interp-{}", std::process::id()));
        let modules = compiled_modules(&home, System::Linux);
        std::fs::create_dir_all(&modules).unwrap();
        for name in [
            "_dbm.cpython-311-x86_64-linux-gnu.so",
            "_ssl.cpython-311-x86_64-linux-gnu.so",
        ] {
            std::fs::write(modules.join(name), b"x").unwrap();
        }
        assert_eq!(
            refused_modules_removed(&home, System::Linux).unwrap(),
            vec!["_dbm.cpython-311-x86_64-linux-gnu.so".to_string()]
        );
        assert!(modules
            .join("_ssl.cpython-311-x86_64-linux-gnu.so")
            .exists());
        assert_eq!(
            refused_modules_removed(&home, System::Linux).unwrap(),
            Vec::<String>::new(),
            "a second run"
        );
        std::fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn unequal_files_are_named() {
        let home = std::env::temp_dir().join(format!("mm-launcher-sums-{}", std::process::id()));
        std::fs::create_dir_all(home.join("bin")).unwrap();
        std::fs::write(home.join("bin/python3.11"), b"not the interpreter").unwrap();
        let linux = target(System::Linux, Processor::X86_64).unwrap();
        assert_eq!(
            unequal_files(&home, linux),
            vec![
                "bin/python3.11".to_string(),
                "lib/libpython3.11.so.1.0".to_string()
            ]
        );
        std::fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn folders_per_system() {
        let home = Path::new("h");
        assert_eq!(compiled_modules(home, System::Windows), Path::new("h/DLLs"));
        assert_eq!(
            compiled_modules(home, System::MacOs),
            Path::new("h/lib/python3.11/lib-dynload")
        );
    }
}
