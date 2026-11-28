//! What the launcher installs on each system: every pinned download with its sha256, the Python lock, where the
//! interpreter and the environment's Python live, and what we may claim about the platform (its tier).
//!
//! The pins shared with setup.sh (uv, Python, llama.cpp on Linux) must stay equal to its own: a test reads setup.sh
//! and compares, until setup.sh becomes two lines that run this program (docs/setup.md, "Platforms").

/// `command` started without a console window of its own: on Windows a console program started from a windowed one
/// (the companion) opens a new console window unless asked not to (CREATE_NO_WINDOW). Only for programs whose output
/// is read or not shown; elsewhere nothing changes.
pub fn quiet(command: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

/// The operating system this program was built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum System {
    Linux,
    Windows,
    MacOs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Processor {
    X86_64,
    Aarch64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Archive {
    TarGz,
    Zip,
    /// A single file, used as it is.
    Plain,
}

/// A file fetched from its maker's own address and refused unless its sha256 is this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Download {
    pub url: &'static str,
    pub sha256: &'static str,
    pub archive: Archive,
}

/// A file of the interpreter (relative to its folder, with `/`) and its sha256 in the publisher's archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpreterFile {
    pub file: &'static str,
    pub sha256: &'static str,
}

/// What we may say about a platform (the platform plan's three tiers), shown by `check` and kept with each object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Linux with an NVIDIA card: every published number was measured here.
    Measured,
    /// The regression set and a batch ran on real hardware of this kind; no sealed run.
    Tried,
    /// Objects were made here and pass every check; the published numbers were measured on Linux.
    Supported,
    /// It installs and passes the small-model smoke test; no object was made on it. Not announced.
    NotSupportedYet,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Measured => "measured",
            Tier::Tried => "tried",
            Tier::Supported => "supported",
            Tier::NotSupportedYet => "not supported yet",
        }
    }

    /// What the tier means, as `check` says it (and every object's record will).
    pub fn words(self) -> &'static str {
        match self {
            Tier::Measured => "measured (every published number was made on this kind of PC)",
            Tier::Tried => "tried (the regression set ran here; no sealed run)",
            Tier::Supported => {
                "supported (objects made here on a 12 GB NVIDIA card; the published numbers were \
                                measured on Linux)"
            }
            Tier::NotSupportedYet => "not supported yet (installs and passes the small-model test)",
        }
    }
}

/// One hashed lock of the Python environment (python/locks/, made by tools/ci/locks.py).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lock {
    pub name: &'static str,
    /// Relative to the repository, with `/`.
    pub file: &'static str,
    pub system: System,
    pub processor: Processor,
    /// Why it may never be installed, when the licence check refuses its wheels.
    pub refused: Option<&'static str>,
}

pub const LOCKS: &[Lock] = &[
    Lock {
        name: "linux-x86_64-cuda128",
        file: "python/requirements.lock",
        system: System::Linux,
        processor: Processor::X86_64,
        refused: None,
    },
    Lock {
        name: "linux-x86_64-cpu",
        file: "python/locks/linux-x86_64-cpu.lock",
        system: System::Linux,
        processor: Processor::X86_64,
        refused: None,
    },
    Lock {
        name: "linux-x86_64-rocm71",
        file: "python/locks/linux-x86_64-rocm71.lock",
        system: System::Linux,
        processor: Processor::X86_64,
        refused: Some("its wheels bundle LGPL and GPL libraries (libnuma, libelf, CHOLMOD); the licence check refuses it"),
    },
    Lock {
        name: "windows-x86_64-cuda128",
        file: "python/locks/windows-x86_64-cuda128.lock",
        system: System::Windows,
        processor: Processor::X86_64,
        refused: None,
    },
    Lock {
        name: "macos-arm64",
        file: "python/locks/macos-arm64.lock",
        system: System::MacOs,
        processor: Processor::Aarch64,
        refused: None,
    },
];

pub const UV_VERSION: &str = "0.13.5";
/// The CPython uv installs for the environment (what the pipeline is tried on), from the standalone builds of this
/// release, whose sums the pinned uv carries.
pub const PYTHON_VERSION: &str = "3.11.14";
pub const PYTHON_BUILD: &str = "20260211";
/// The question judge's server: the tag it was measured with (another build answers differently).
pub const LLAMA_TAG: &str = "b11600";

/// Everything the launcher pins for one system and processor.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub system: System,
    pub processor: Processor,
    /// The lock installed unless `--lock` names another of this system's; None: not supported here.
    pub default_lock: Option<&'static str>,
    pub uv: Download,
    /// llama.cpp's CPU release archive, when there is one for this target.
    pub llama: Option<Download>,
    /// llama.cpp's licence text from the same tag's source, where its archive carries none (Windows).
    pub llama_licence: Option<Download>,
    /// uv's folder name for the interpreter under `.tools/python`.
    pub interpreter_folder: &'static str,
    /// Files of the interpreter, checked after uv installs it.
    pub interpreter_sums: &'static [InterpreterFile],
    /// The environment's Python, relative to `.venv`.
    pub venv_python: &'static str,
    pub tier: Tier,
}

const LLAMA_LICENCE: Download = Download {
    url: "https://raw.githubusercontent.com/ggml-org/llama.cpp/b11600/LICENSE",
    sha256: "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d",
    archive: Archive::Plain,
};

pub const TARGETS: &[Target] = &[
    Target {
        system: System::Linux,
        processor: Processor::X86_64,
        default_lock: Some("linux-x86_64-cuda128"),
        uv: Download {
            url: "https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-unknown-linux-gnu.tar.gz",
            sha256: "f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f",
            archive: Archive::TarGz,
        },
        llama: Some(Download {
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11600/llama-b11600-bin-ubuntu-x64.tar.gz",
            sha256: "a5ce0dfd864e2acad5922e0d508fe56aaa18298e79327f511efa93b16dcbfbcc",
            archive: Archive::TarGz,
        }),
        llama_licence: None,
        interpreter_folder: "cpython-3.11.14-linux-x86_64-gnu",
        interpreter_sums: &[
            InterpreterFile { file: "bin/python3.11", sha256: "6ff97f602038740073dca96714310a30e303332326268e0f1bb2767edc820944" },
            InterpreterFile { file: "lib/libpython3.11.so.1.0", sha256: "7f12c12f3a1d15355a4855f22cd12086ca60620d59db6b58df1a54df95040f18" },
        ],
        venv_python: "bin/python",
        tier: Tier::Measured,
    },
    // setup.sh pins uv for Linux on arm64 too; there is no lock for it, so `check` says it is not supported.
    Target {
        system: System::Linux,
        processor: Processor::Aarch64,
        default_lock: None,
        uv: Download {
            url: "https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-unknown-linux-gnu.tar.gz",
            sha256: "4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403",
            archive: Archive::TarGz,
        },
        llama: None,
        llama_licence: None,
        interpreter_folder: "cpython-3.11.14-linux-aarch64-gnu",
        interpreter_sums: &[],
        venv_python: "bin/python",
        tier: Tier::NotSupportedYet,
    },
    Target {
        system: System::Windows,
        processor: Processor::X86_64,
        default_lock: Some("windows-x86_64-cuda128"),
        uv: Download {
            url: "https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip",
            sha256: "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a",
            archive: Archive::Zip,
        },
        llama: Some(Download {
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11600/llama-b11600-bin-win-cpu-x64.zip",
            sha256: "ea688fd5cd702f0f1a3703d1b5ba098a9aea21a87221c8d7b2c50245e3beabdf",
            archive: Archive::Zip,
        }),
        llama_licence: Some(LLAMA_LICENCE),
        interpreter_folder: "cpython-3.11.14-windows-x86_64-none",
        interpreter_sums: &[
            InterpreterFile { file: "python.exe", sha256: "77ce191aa5d30b17288a3d6adaf83813154998b8de82b82485d393d11ce47c58" },
            InterpreterFile { file: "python311.dll", sha256: "2401df7182ab25100deab6632cdd4029b0809ae9984c0bb6f44c8510df6b7ebd" },
        ],
        venv_python: "Scripts/python.exe",
        tier: Tier::Supported,
    },
    Target {
        system: System::MacOs,
        processor: Processor::Aarch64,
        default_lock: Some("macos-arm64"),
        uv: Download {
            url: "https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-apple-darwin.tar.gz",
            sha256: "743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73",
            archive: Archive::TarGz,
        },
        llama: Some(Download {
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11600/llama-b11600-bin-macos-arm64.tar.gz",
            sha256: "a9c10e58e42603d67a3ad3cbe991d420e31d350937e11775db55011e25536668",
            archive: Archive::TarGz,
        }),
        llama_licence: None,
        interpreter_folder: "cpython-3.11.14-macos-aarch64-none",
        interpreter_sums: &[
            InterpreterFile { file: "bin/python3.11", sha256: "e6eedfbc57422e986a0e3b24b18ee45764b809ff1385d3af42e9c22b5bd00de5" },
            InterpreterFile { file: "lib/libpython3.11.dylib", sha256: "44f6c26021051a7168e9b149696f9c35a49e16f8b6c841ebd7a9c8994a7d9f2b" },
        ],
        venv_python: "bin/python",
        tier: Tier::NotSupportedYet,
    },
];

/// The system and processor this program was built for.
pub fn here() -> (System, Processor) {
    let system = if cfg!(windows) {
        System::Windows
    } else if cfg!(target_os = "macos") {
        System::MacOs
    } else {
        System::Linux
    };
    let processor = if cfg!(target_arch = "aarch64") {
        Processor::Aarch64
    } else {
        Processor::X86_64
    };
    (system, processor)
}

pub fn target(system: System, processor: Processor) -> Option<&'static Target> {
    TARGETS
        .iter()
        .find(|t| t.system == system && t.processor == processor)
}

/// Why a lock was not chosen.
#[derive(Debug, PartialEq, Eq)]
pub enum LockRefusal {
    Unknown(String),
    OtherSystem(&'static str),
    Refused(&'static str, &'static str),
    NoneForThisPc,
}

impl std::fmt::Display for LockRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockRefusal::Unknown(name) => {
                let known: Vec<&str> = LOCKS.iter().map(|l| l.name).collect();
                write!(
                    f,
                    "there is no lock named {name} (known: {})",
                    known.join(", ")
                )
            }
            LockRefusal::OtherSystem(name) => {
                write!(f, "the lock {name} is for another system or processor")
            }
            LockRefusal::Refused(name, why) => {
                write!(f, "the lock {name} is never installed: {why}")
            }
            LockRefusal::NoneForThisPc => {
                write!(f, "there is no lock for this system and processor")
            }
        }
    }
}

/// The lock to install: the one `--lock` names (only this system's, never a refused one), else the target's own.
pub fn chosen_lock(target: &Target, asked: Option<&str>) -> Result<&'static Lock, LockRefusal> {
    let name = match asked {
        Some(name) => name,
        None => target.default_lock.ok_or(LockRefusal::NoneForThisPc)?,
    };
    let lock = LOCKS
        .iter()
        .find(|l| l.name == name)
        .ok_or_else(|| LockRefusal::Unknown(name.to_string()))?;
    if lock.system != target.system || lock.processor != target.processor {
        return Err(LockRefusal::OtherSystem(lock.name));
    }
    if let Some(why) = lock.refused {
        return Err(LockRefusal::Refused(lock.name, why));
    }
    Ok(lock)
}

/// A program's file name on this system (`.exe` on Windows).
pub fn program(system: System, stem: &str) -> String {
    if system == System::Windows {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_sha256(text: &str) -> bool {
        text.len() == 64
            && text
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    }

    fn downloads() -> Vec<Download> {
        TARGETS
            .iter()
            .flat_map(|t| [Some(t.uv), t.llama, t.llama_licence])
            .flatten()
            .collect()
    }

    #[test]
    fn every_download_is_https_with_a_sha256_and_never_a_batch_file() {
        for d in downloads() {
            assert!(d.url.starts_with("https://"), "{}", d.url);
            assert!(is_sha256(d.sha256), "{}", d.url);
            for refused in [".cmd", ".bat", ".ps1"] {
                assert!(!d.url.to_ascii_lowercase().ends_with(refused), "{}", d.url);
            }
        }
        for t in TARGETS {
            for f in t.interpreter_sums {
                assert!(is_sha256(f.sha256), "{}", f.file);
                assert!(!f.file.contains('\\'), "{}: written with /", f.file);
            }
            assert!(!t.venv_python.ends_with(".cmd") && !t.venv_python.ends_with(".bat"));
        }
    }

    #[test]
    fn downloads_name_the_pinned_versions_and_their_own_target() {
        for t in TARGETS {
            assert!(
                t.uv.url.contains(&format!("/{UV_VERSION}/")),
                "{}",
                t.uv.url
            );
            let triple = match (t.system, t.processor) {
                (System::Linux, Processor::X86_64) => "x86_64-unknown-linux-gnu",
                (System::Linux, Processor::Aarch64) => "aarch64-unknown-linux-gnu",
                (System::Windows, _) => "x86_64-pc-windows-msvc",
                (System::MacOs, _) => "aarch64-apple-darwin",
            };
            assert!(t.uv.url.contains(triple), "{}", t.uv.url);
            if let Some(llama) = t.llama {
                assert!(
                    llama
                        .url
                        .contains(&format!("/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-")),
                    "{}",
                    llama.url
                );
            }
            if let Some(licence) = t.llama_licence {
                assert!(
                    licence.url.contains(&format!("/{LLAMA_TAG}/")),
                    "{}",
                    licence.url
                );
            }
            assert!(t
                .interpreter_folder
                .starts_with(&format!("cpython-{PYTHON_VERSION}-")));
        }
    }

    #[test]
    fn windows_gets_llama_licence_from_the_source_because_its_zip_has_none() {
        let windows = target(System::Windows, Processor::X86_64).unwrap();
        assert_eq!(windows.llama.unwrap().archive, Archive::Zip);
        assert_eq!(windows.llama_licence.unwrap().archive, Archive::Plain);
    }

    #[test]
    fn every_default_lock_is_known_for_its_own_target_and_not_refused() {
        for t in TARGETS {
            if t.default_lock.is_some() {
                let lock = chosen_lock(t, None).unwrap();
                assert_eq!((lock.system, lock.processor), (t.system, t.processor));
            }
        }
    }

    /// "The interpreter's sums, with no fallback": every target that can be set up checks the interpreter and its
    /// library.
    #[test]
    fn every_target_with_a_lock_checks_the_interpreter_and_its_library() {
        for t in TARGETS.iter().filter(|t| t.default_lock.is_some()) {
            let wanted: [&str; 2] = match t.system {
                System::Linux => ["bin/python3.11", "lib/libpython3.11.so.1.0"],
                System::Windows => ["python.exe", "python311.dll"],
                System::MacOs => ["bin/python3.11", "lib/libpython3.11.dylib"],
            };
            for file in wanted {
                assert!(
                    t.interpreter_sums.iter().any(|f| f.file == file),
                    "{:?} {:?}: {file}",
                    t.system,
                    t.processor
                );
            }
        }
    }

    #[test]
    fn lock_files_are_in_the_repository() {
        let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for lock in LOCKS {
            assert!(repository.join(lock.file).is_file(), "{}", lock.file);
        }
    }

    #[test]
    fn a_refused_lock_can_never_be_chosen() {
        let linux = target(System::Linux, Processor::X86_64).unwrap();
        let refused = chosen_lock(linux, Some("linux-x86_64-rocm71")).unwrap_err();
        assert!(matches!(
            refused,
            LockRefusal::Refused("linux-x86_64-rocm71", _)
        ));
        assert!(refused.to_string().contains("never installed"));
    }

    #[test]
    fn lock_choice_is_limited_to_this_system() {
        let linux = target(System::Linux, Processor::X86_64).unwrap();
        assert_eq!(
            chosen_lock(linux, Some("linux-x86_64-cpu")).unwrap().name,
            "linux-x86_64-cpu"
        );
        assert_eq!(
            chosen_lock(linux, None).unwrap().file,
            "python/requirements.lock"
        );
        assert_eq!(
            chosen_lock(linux, Some("windows-x86_64-cuda128")),
            Err(LockRefusal::OtherSystem("windows-x86_64-cuda128"))
        );
        assert!(matches!(
            chosen_lock(linux, Some("nonsense")),
            Err(LockRefusal::Unknown(_))
        ));
        let arm = target(System::Linux, Processor::Aarch64).unwrap();
        assert_eq!(chosen_lock(arm, None), Err(LockRefusal::NoneForThisPc));
    }

    #[test]
    fn program_names() {
        assert_eq!(program(System::Windows, "llama-server"), "llama-server.exe");
        assert_eq!(program(System::Linux, "llama-server"), "llama-server");
    }

    #[test]
    fn this_build_has_a_target() {
        let (system, processor) = here();
        assert!(target(system, processor).is_some());
    }
}
