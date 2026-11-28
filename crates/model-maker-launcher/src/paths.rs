//! Real paths as Windows' own programs write them. `std::fs::canonicalize` gives a verbatim `\\?\C:\...` path there;
//! handed on (the Python side's PYTHONPATH, its `__file__`), Windows refuses any `..` later joined onto it (the card
//! test of 28 Nov 2026: WinError 123 in the ledger). So a verbatim path is written the plain way whenever that names
//! the same file, and kept verbatim only where it must be: too long for the plain form, or a name that form reads
//! differently (a device name such as `NUL`, a trailing dot or space, a `.` or `..` part).

use std::path::{Path, PathBuf};

const VERBATIM: &str = r"\\?\";
/// The plain form's limit (MAX_PATH, 260 with its closing NUL).
const PLAIN_LONGEST: usize = 259;
/// Names the plain form reads as devices, with or without an extension (`nul.txt` too).
const DEVICES: [&str; 4] = ["CON", "PRN", "AUX", "NUL"];
const NUMBERED_DEVICES: [&str; 2] = ["COM", "LPT"];
const REFUSED_CHARACTERS: [char; 8] = ['<', '>', ':', '"', '/', '|', '?', '*'];

/// The path with its links resolved, written the plain way when it can be (see the module's note).
pub fn real(path: &Path) -> PathBuf {
    simplified(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// `\\?\C:\folder` as `C:\folder` when the plain form names the same file; any other path as it is (network paths
/// `\\?\UNC\...` too: nothing here is installed on one).
pub fn simplified(path: PathBuf) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path;
    };
    match text.strip_prefix(VERBATIM) {
        Some(plain) if plain_names_the_same(plain) => PathBuf::from(plain),
        _ => path,
    }
}

/// `C:\a\b`: a drive, then parts the plain form reads exactly as written, within its length.
fn plain_names_the_same(plain: &str) -> bool {
    let bytes = plain.as_bytes();
    let on_a_drive =
        bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'\\';
    if !on_a_drive || plain.len() > PLAIN_LONGEST {
        return false;
    }
    let rest = &plain[3..];
    rest.is_empty() || rest.split('\\').all(plain_part)
}

fn plain_part(part: &str) -> bool {
    !part.is_empty()
        && part != "."
        && part != ".."
        && !part.ends_with(['.', ' '])
        && !part
            .chars()
            .any(|c| c.is_control() || REFUSED_CHARACTERS.contains(&c))
        && !is_device(part)
}

fn is_device(part: &str) -> bool {
    let stem = part
        .split('.')
        .next()
        .unwrap_or(part)
        .trim_end()
        .to_ascii_uppercase();
    if DEVICES.contains(&stem.as_str()) || stem == "CONIN$" || stem == "CONOUT$" {
        return true;
    }
    NUMBERED_DEVICES.iter().any(|name| {
        stem.strip_prefix(name).is_some_and(|number| {
            matches!(
                number,
                "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple(text: &str) -> String {
        simplified(PathBuf::from(text))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn a_short_verbatim_drive_path_is_written_the_plain_way() {
        assert_eq!(
            simple(r"\\?\C:\Programs\model-maker-0.4.0"),
            r"C:\Programs\model-maker-0.4.0"
        );
        assert_eq!(simple(r"\\?\D:\"), r"D:\");
        assert_eq!(simple(r"\\?\C:\a\file.name.txt"), r"C:\a\file.name.txt");
        assert_eq!(
            simple(r"\\?\C:\a\console\nullable"),
            r"C:\a\console\nullable"
        );
    }

    #[test]
    fn a_verbatim_path_the_plain_form_would_read_differently_is_kept() {
        for kept in [
            r"\\?\C:\a\NUL",
            r"\\?\C:\a\nul.txt",
            r"\\?\C:\a\COM1",
            r"\\?\C:\a\lpt9.log",
            r"\\?\C:\a\CONOUT$",
            r"\\?\C:\a\ends-with-a-dot.",
            r"\\?\C:\a\ends-with-a-space ",
            r"\\?\C:\a\..\b",
            r"\\?\C:\a\.\b",
            r"\\?\C:\a\\b",
            r"\\?\C:\a/b",
            r"\\?\C:\a\b?",
            r"\\?\C:",
            r"\\?\UNC\server\share\folder",
            r"\\?\Volume{12345678-1234-1234-1234-123456789abc}\folder",
        ] {
            assert_eq!(simple(kept), kept, "{kept}");
        }
    }

    #[test]
    fn a_verbatim_path_too_long_for_the_plain_form_is_kept() {
        let longest = format!(r"C:\{}", "a".repeat(PLAIN_LONGEST - 3));
        assert_eq!(simple(&format!(r"\\?\{longest}")), longest);
        let too_long = format!(r"\\?\C:\{}", "a".repeat(PLAIN_LONGEST - 2));
        assert_eq!(simple(&too_long), too_long);
    }

    #[test]
    fn other_paths_are_left_as_they_are() {
        for path in [
            "/srv/model-maker",
            r"C:\already\plain",
            r"\\server\share",
            "relative",
        ] {
            assert_eq!(simple(path), path);
        }
    }

    #[test]
    fn the_launcher_takes_every_real_path_through_this_module() {
        // The program's folder and the repository came straight from canonicalize until 28 Nov 2026.
        let sources = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut direct = Vec::new();
        for entry in std::fs::read_dir(&sources).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "rs") && !path.ends_with("paths.rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                if text.contains("canonicalize(") {
                    direct.push(path.file_name().unwrap().to_string_lossy().into_owned());
                }
            }
        }
        direct.sort();
        assert!(
            direct.is_empty(),
            "canonicalize called outside paths.rs: {direct:?}"
        );
    }
}
