//! The two programs besides Python: the question judge's server (llama.cpp's own CPU release, at the tag it was
//! measured with) and model-maker-simplify (ours, prebuilt beside this program in a release).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::download;
use crate::platform::{self, Archive, System, Target};

/// The server's folder under `.tools` (the same name setup.sh uses).
pub fn judge_folder(tools: &Path) -> PathBuf {
    tools.join(format!("llama-{}", platform::LLAMA_TAG))
}

pub fn judge_server(tools: &Path, system: System) -> PathBuf {
    judge_folder(tools).join(platform::program(system, "llama-server"))
}

/// A shared library, by any system's names.
pub fn is_library(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".dll")
        || lower.ends_with(".dylib")
        || lower.ends_with(".so")
        || lower.contains(".so.")
}

/// The licence of a library llama.cpp's release archives carry, by its name; None for one nobody has read.
/// Its own libraries are MIT (its LICENSE); the Windows zip also carries LLVM's OpenMP runtime.
pub fn judge_library_licence(name: &str) -> Option<&'static str> {
    const OWN: &[&str] = &["libggml", "libllama", "libmtmd", "ggml", "llama", "mtmd"];
    let lower = name.to_ascii_lowercase();
    if lower.starts_with("libomp") {
        return Some("Apache-2.0 WITH LLVM-exception (LLVM's OpenMP runtime, its LICENSE-LLVM-OpenMP beside it)");
    }
    OWN.iter()
        .any(|own| lower.starts_with(own))
        .then_some("MIT (llama.cpp's LICENSE)")
}

/// The libraries in the folder that are not in the table: each one refuses the archive.
pub fn unknown_libraries(folder: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut unknown: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| is_library(name) && judge_library_licence(name).is_none())
        .collect();
    unknown.sort();
    unknown
}

fn is_mit_licence(file: &Path) -> bool {
    std::fs::read_to_string(file).is_ok_and(|text| text.starts_with("MIT License"))
}

/// Why the judge is off, in words (the judge is optional: setup carries on).
#[derive(Debug, PartialEq, Eq)]
pub struct JudgeOff(pub String);

/// The server unpacked whole into `.tools/llama-<tag>` (through a `.part` folder, so a broken try leaves nothing),
/// its LICENSE beside it (from the tag's source where the archive has none), every library in the licence table.
pub fn installed_judge(tools: &Path, target: &Target) -> Result<String, JudgeOff> {
    let tag = platform::LLAMA_TAG;
    let folder = judge_folder(tools);
    if judge_server(tools, target.system).is_file() {
        return Ok(format!("llama.cpp {tag}: already in .tools/llama-{tag}"));
    }
    let Some(archive) = target.llama else {
        return Err(JudgeOff(format!(
            "llama.cpp {tag}: no pinned archive for this system"
        )));
    };
    let part = tools.join(format!("llama-{tag}.part"));
    let _ = std::fs::remove_dir_all(&part);
    let result = unpacked_judge(tools, target, &archive, &part);
    match result {
        Ok(()) => {
            std::fs::rename(&part, &folder)
                .map_err(|e| JudgeOff(format!("llama.cpp {tag}: {e}")))?;
            Ok(format!(
                "llama.cpp {tag} (MIT): installed in .tools/llama-{tag}"
            ))
        }
        Err(why) => {
            let _ = std::fs::remove_dir_all(&part);
            Err(JudgeOff(format!(
                "llama.cpp {tag}: optional, not used ({why}): the question judge is off"
            )))
        }
    }
}

fn unpacked_judge(
    tools: &Path,
    target: &Target,
    archive: &platform::Download,
    part: &Path,
) -> Result<(), String> {
    let file = tools.join(if archive.archive == Archive::Zip {
        "llama.zip"
    } else {
        "llama.tar.gz"
    });
    download::fetched(archive, &file).map_err(|e| e.to_string())?;
    let strip = usize::from(archive.archive == Archive::TarGz);
    let unpacked = download::unpacked(&file, archive.archive, part, strip);
    let _ = std::fs::remove_file(&file);
    unpacked.map_err(|e| e.to_string())?;
    if let Some(licence) = target.llama_licence {
        download::fetched(&licence, &part.join("LICENSE")).map_err(|e| e.to_string())?;
    }
    if !part
        .join(platform::program(target.system, "llama-server"))
        .is_file()
    {
        return Err("the archive has no llama-server".into());
    }
    if !is_mit_licence(&part.join("LICENSE")) {
        return Err("no MIT LICENSE of llama.cpp beside it".into());
    }
    let unknown = unknown_libraries(part);
    if !unknown.is_empty() {
        return Err(format!(
            "libraries nobody has read the licence of: {}",
            unknown.join(", ")
        ));
    }
    Ok(())
}

/// `--without-question-judge` after a setup with it: the server removed (its model stays in models/).
pub fn removed_judge(tools: &Path) -> Option<String> {
    let folder = judge_folder(tools);
    std::fs::remove_dir_all(&folder).ok().map(|_| {
        format!("llama.cpp {}: removed; the question judge is off (its model stays in models/qwen3.5-9b-q4)",
                platform::LLAMA_TAG)
    })
}

/// model-maker-simplify: the one that came beside this program (a release), else one built in the repository.
pub fn simplify_tool(
    repository: &Path,
    system: System,
    program_folder: Option<&Path>,
) -> Option<PathBuf> {
    let name = platform::program(system, "model-maker-simplify");
    let beside = program_folder.map(|folder| folder.join(&name));
    let built = repository.join("target").join("release").join(&name);
    beside
        .into_iter()
        .chain([built])
        .find(|path| path.is_file())
}

/// The folder this program is in.
pub fn program_folder() -> Option<PathBuf> {
    let program = std::env::current_exe().ok()?;
    crate::paths::real(&program).parent().map(Path::to_path_buf)
}

/// Built here when no prebuilt one came with this program: needs Rust (rust-toolchain.toml pins it) and a C++
/// compiler (meshoptimizer).
pub fn built_simplify(repository: &Path) -> Result<(), String> {
    let status = Command::new("cargo")
        .args(["build", "--release", "--locked", "--quiet", "-p", "model-maker-simplify"])
        .current_dir(repository)
        .status()
        .map_err(|_| {
            "model-maker-simplify did not come with this program and there is no Rust here to build it: use the \
             release archive for your system (it has the tool), or install Rust (https://rustup.rs) and a C++ \
             compiler and run setup again"
                .to_string()
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "building model-maker-simplify failed ({status}); it needs a C++ compiler"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary(name: &str) -> PathBuf {
        let folder =
            std::env::temp_dir().join(format!("mm-launcher-tools-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    #[test]
    fn every_library_of_the_three_archives_is_in_the_table() {
        // Read from the b11600 archives' file lists (Linux, macOS, Windows).
        for name in [
            "libggml-base.so.0.26.0",
            "libggml-cpu-zen4.so",
            "libllama.so.0.6.0",
            "libmtmd.so.0",
            "libllama-server-impl.so",
            "libggml-metal.0.26.0.dylib",
            "libllama-common.0.6.0.dylib",
            "ggml-base.dll",
            "ggml-cpu-sapphirerapids.dll",
            "llama.dll",
            "llama-server-impl.dll",
            "mtmd.dll",
            "libomp.dll",
        ] {
            assert!(is_library(name), "{name}");
            assert!(judge_library_licence(name).is_some(), "{name}");
        }
        assert!(judge_library_licence("libomp.dll")
            .unwrap()
            .contains("LLVM-exception"));
        for unknown in ["msvcp140.dll", "libcurl.so.4", "vcruntime140.dll"] {
            assert!(judge_library_licence(unknown).is_none(), "{unknown}");
        }
        assert!(!is_library("llama-server.exe") && !is_library("LICENSE"));
    }

    #[test]
    fn an_unknown_library_is_named() {
        let folder = temporary("unknown");
        for name in ["ggml.dll", "libomp.dll", "extra.dll", "llama-server.exe"] {
            std::fs::write(folder.join(name), b"x").unwrap();
        }
        assert_eq!(unknown_libraries(&folder), vec!["extra.dll".to_string()]);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn an_installed_server_is_kept_and_removal_says_so() {
        let tools = temporary("judge");
        let system = platform::here().0;
        let server = judge_server(&tools, system);
        std::fs::create_dir_all(server.parent().unwrap()).unwrap();
        std::fs::write(&server, b"x").unwrap();
        let target = platform::target(system, platform::here().1).unwrap();
        assert!(installed_judge(&tools, target).unwrap().contains("already"));
        assert!(removed_judge(&tools).unwrap().contains("removed"));
        assert!(removed_judge(&tools).is_none());
        std::fs::remove_dir_all(&tools).unwrap();
    }

    #[test]
    fn the_prebuilt_tool_beside_the_program_comes_first() {
        let repository = temporary("simplify");
        let release = repository.join("target").join("release");
        std::fs::create_dir_all(&release).unwrap();
        let name = platform::program(System::Linux, "model-maker-simplify");
        assert_eq!(simplify_tool(&repository, System::Linux, None), None);
        std::fs::write(release.join(&name), b"x").unwrap();
        assert_eq!(
            simplify_tool(&repository, System::Linux, None),
            Some(release.join(&name))
        );
        let beside = repository.join("bin");
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::write(beside.join(&name), b"x").unwrap();
        assert_eq!(
            simplify_tool(&repository, System::Linux, Some(&beside)),
            Some(beside.join(&name))
        );
        assert!(
            simplify_tool(&repository, System::Windows, Some(&beside)).is_none(),
            ".exe on Windows"
        );
        std::fs::remove_dir_all(&repository).unwrap();
    }

    /// The real archive of this system, fetched and unpacked (needs the network: `cargo test -- --ignored`).
    #[test]
    #[ignore]
    fn the_real_archive_installs_and_passes_its_checks() {
        let tools = temporary("real");
        let (system, processor) = platform::here();
        let target = platform::target(system, processor).unwrap();
        let said = installed_judge(&tools, target).unwrap();
        assert!(said.contains("installed"), "{said}");
        assert!(judge_server(&tools, system).is_file());
        assert!(unknown_libraries(&judge_folder(&tools)).is_empty());
        std::fs::remove_dir_all(&tools).unwrap();
    }
}
