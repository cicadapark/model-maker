//! `model-maker check`: what this PC has against what model-maker needs, as a table; nothing is changed.
//!
//! The facts are gathered from the system (`gather`) and judged by a pure function (`rows`), so the judging is
//! tested with recorded facts from every kind of PC. The limits are setup.sh's own (a test compares them).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::platform::{self, Lock, Processor, System, Target};

/// The picture and the views models together.
pub const GPU_GOOD_MIB: u64 = 16000;
/// With the models loaded one at a time: "try it".
pub const GPU_TRY_MIB: u64 = 12000;
pub const CUDA_MAJOR: u32 = 12;
/// The models' files, about 57 GB (setup with --skip-models does not need room for them).
pub const MODELS_GB: u64 = 57;
/// Loading the picture model peaked at 32 GB resident, much of it the weights' files mapped into memory.
pub const MEMORY_GOOD_GIB: u64 = 24;
/// With a card under GPU_GOOD_MIB the view painter's blocks take turns: 0.3.3 peaked at 25.5 GB there, and less
/// was not tried (docs/results.md).
pub const MEMORY_GOOD_GIB_SMALL_CARD: u64 = 32;
pub const MEMORY_LEAST_GIB: u64 = 16;
/// Model weights about 57 GB, the Python environment about 7.5 GB, builds.
pub const DISK_GB: u64 = 77;
/// On Windows the environment is about 5 GB and uv's download cache as much again while it installs (measured
/// in the Windows test, 25 Nov 2026).
pub const WINDOWS_EXTRA_DISK_GB: u64 = 5;
pub const DISK_LEAST_GB: u64 = 5;
/// With long paths off, Windows refuses a path over 259 characters; the longest installed path is 168 characters
/// below `.tools\python` (the interpreter's own test data), so the repository's own path may be 75 at most.
pub const WINDOWS_REPOSITORY_PATH_MOST: usize = 75;
pub const TESSERACT_MAJOR: u32 = 5;
/// The C++ runtime torch's and llama.cpp's Windows libraries need; a fresh Windows has none of it.
pub const WINDOWS_CXX_RUNTIME: &[&str] =
    &["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    pub name: String,
    pub memory_mib: u64,
    pub driver: String,
}

/// What `nvidia-smi` said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cards {
    NoTool,
    ToolFailed,
    Found(Vec<Card>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PythonFound {
    Environment(String),
    System(String),
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tesseract {
    None,
    Found {
        version: Option<String>,
        english: bool,
    },
}

/// Everything `rows` judges, gathered from the PC (or recorded, in the tests).
#[derive(Clone, Debug)]
pub struct Facts {
    pub system: System,
    pub processor: Processor,
    pub wsl: bool,
    pub cards: Cards,
    pub cuda: Option<String>,
    pub memory_gib: u64,
    pub free_disk_gb: u64,
    pub models_gb: u64,
    pub python: PythonFound,
    pub git: Option<String>,
    pub tesseract: Tesseract,
    pub judge_installed: bool,
    /// Setup's own record (setup::DONE_FILE).
    pub setup_done: crate::setup::Done,
    pub simplify_found: bool,
    /// Windows only: the runtime's files missing from System32.
    pub cxx_runtime_missing: Vec<&'static str>,
    /// Windows only: whether long paths are on (None elsewhere, or when it could not be read).
    pub long_paths: Option<bool>,
    pub repository_path_chars: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Ok(String),
    /// Needed and not there: counted, and `check` exits 1.
    Missing(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub check: &'static str,
    pub found: String,
    pub verdict: Verdict,
}

fn ok(check: &'static str, found: impl Into<String>, words: impl Into<String>) -> Row {
    Row {
        check,
        found: found.into(),
        verdict: Verdict::Ok(words.into()),
    }
}

fn missing(check: &'static str, found: impl Into<String>, needed: impl Into<String>) -> Row {
    Row {
        check,
        found: found.into(),
        verdict: Verdict::Missing(needed.into()),
    }
}

/// The table's rows for these facts. `needs_card`: false for a processor-only run (CI, the test machine), which
/// needs neither a card nor a make's memory.
pub fn rows(
    facts: &Facts,
    target: Option<&Target>,
    lock: Option<&Lock>,
    needs_card: bool,
) -> Vec<Row> {
    let mut rows = vec![system_row(facts), processor_row(facts, target, lock)];
    rows.extend(card_rows(facts, needs_card));
    rows.push(memory_row(facts, needs_card));
    rows.push(disk_row(facts));
    if facts.system == System::Windows {
        rows.push(cxx_runtime_row(facts));
        rows.push(path_row(facts));
    }
    rows.push(python_row(&facts.python));
    rows.push(simplify_row(facts.simplify_found));
    rows.push(match &facts.git {
        Some(version) => ok("git", version.clone(), "ok"),
        None => missing("git", "none", git_words(facts.system)),
    });
    rows.push(tesseract_row(&facts.tesseract, facts.system));
    rows.push(judge_row(facts.judge_installed));
    rows.push(setup_row(facts.setup_done));
    if let Some(target) = target {
        rows.push(ok("Platform", target.tier.name(), target.tier.words()));
    }
    rows.push(ok(
        "This program",
        "unsigned build",
        "check the archive's sha256 against the published one",
    ));
    rows
}

fn system_row(facts: &Facts) -> Row {
    match facts.system {
        System::Linux if facts.wsl => {
            ok("System", "Linux (WSL2)", "should work, not yet tried by us")
        }
        System::Linux => ok("System", "Linux", "ok"),
        System::Windows => ok("System", "Windows", "ok"),
        System::MacOs => ok(
            "System",
            "macOS",
            "not supported yet (installs and passes the small-model test)",
        ),
    }
}

fn processor_row(facts: &Facts, target: Option<&Target>, lock: Option<&Lock>) -> Row {
    let found = match facts.processor {
        Processor::X86_64 => "x86_64",
        Processor::Aarch64 => "arm64",
    };
    match (target, lock) {
        (Some(_), Some(lock)) => ok("Processor", found, format!("ok (lock {})", lock.name)),
        _ => missing("Processor", found, "a system and processor model-maker has a lock for (Linux and Windows on \
                                           x86_64, macOS on Apple Silicon); on Linux arm64 there is none yet"),
    }
}

fn card_rows(facts: &Facts, needs_card: bool) -> Vec<Row> {
    const CHECK: &str = "Graphics card";
    if facts.system == System::MacOs {
        return vec![ok(
            CHECK,
            format!("Apple Silicon, {} GB shared", facts.memory_gib),
            "not supported yet: the card and the computer share this memory",
        )];
    }
    if !needs_card {
        return vec![ok(
            CHECK,
            "not needed",
            "this lock runs on the processor only (slow; for tests)",
        )];
    }
    let cards = match &facts.cards {
        Cards::NoTool => {
            return vec![missing(
                CHECK,
                "no nvidia-smi",
                "an NVIDIA card and its driver",
            )]
        }
        Cards::ToolFailed => {
            return vec![missing(
                CHECK,
                "nvidia-smi sees no card",
                "a working NVIDIA driver",
            )]
        }
        Cards::Found(cards) => cards,
    };
    let Some(best) = cards.iter().max_by_key(|c| c.memory_mib) else {
        return vec![missing(
            CHECK,
            "nvidia-smi sees no card",
            "a working NVIDIA driver",
        )];
    };
    let found = format!("{}, {} GB", best.name, best.memory_mib / 1024);
    let card = if best.memory_mib >= GPU_GOOD_MIB {
        ok(CHECK, found, "ok")
    } else if best.memory_mib >= GPU_TRY_MIB {
        ok(
            CHECK,
            found,
            "ok (under 16 GB the view painter's blocks take turns: docs/setup.md)",
        )
    } else {
        missing(CHECK, found, "a card with at least 12 GB")
    };
    let cuda = match facts.cuda.as_deref() {
        None => missing(
            "CUDA",
            "not reported",
            format!("a driver with CUDA {CUDA_MAJOR}.x"),
        ),
        Some(version) if major(version).is_some_and(|m| m >= CUDA_MAJOR) => {
            ok("CUDA", format!("{version} (driver {})", best.driver), "ok")
        }
        Some(version) => missing(
            "CUDA",
            version,
            format!("a driver with CUDA {CUDA_MAJOR}.x or newer"),
        ),
    };
    vec![card, cuda]
}

fn major(version: &str) -> Option<u32> {
    version.split('.').next()?.trim().parse().ok()
}

/// The memory advised for a make: more with a card under GPU_GOOD_MIB (setup.sh's check_memory).
fn memory_good_gib(cards: &Cards) -> u64 {
    let best_mib = match cards {
        Cards::Found(found) => found.iter().map(|c| c.memory_mib).max().unwrap_or(0),
        Cards::NoTool | Cards::ToolFailed => 0,
    };
    if best_mib > 0 && best_mib < GPU_GOOD_MIB {
        MEMORY_GOOD_GIB_SMALL_CARD
    } else {
        MEMORY_GOOD_GIB
    }
}

fn memory_row(facts: &Facts, for_making: bool) -> Row {
    let found = format!("{} GB", facts.memory_gib);
    let good = memory_good_gib(&facts.cards);
    if !for_making && facts.memory_gib < MEMORY_LEAST_GIB {
        ok(
            "Memory",
            found,
            format!("enough for tests on the processor; a make needs {MEMORY_LEAST_GIB} GB"),
        )
    } else if facts.memory_gib >= good {
        ok("Memory", found, "ok")
    } else if facts.memory_gib >= MEMORY_LEAST_GIB {
        ok(
            "Memory",
            found,
            format!("try it ({good} GB or more is advised)"),
        )
    } else {
        missing(
            "Memory",
            found,
            format!("at least {MEMORY_LEAST_GIB} GB ({good} GB or more is advised)"),
        )
    }
}

/// What a full setup still needs here: what an earlier run already put in models/ counts as done.
pub fn disk_needed_gb(system: System, models_gb: u64) -> u64 {
    let extra = if system == System::Windows {
        WINDOWS_EXTRA_DISK_GB
    } else {
        0
    };
    (DISK_GB + extra)
        .saturating_sub(models_gb)
        .max(DISK_LEAST_GB)
}

fn disk_row(facts: &Facts) -> Row {
    let need = disk_needed_gb(facts.system, facts.models_gb);
    let found = format!("{} GB", facts.free_disk_gb);
    if facts.free_disk_gb >= need {
        ok("Free disk here", found, "ok")
    } else {
        missing(
            "Free disk here",
            found,
            format!("at least {need} GB free in this folder"),
        )
    }
}

fn cxx_runtime_row(facts: &Facts) -> Row {
    const CHECK: &str = "C++ runtime";
    if facts.cxx_runtime_missing.is_empty() {
        ok(CHECK, "Visual C++ runtime", "ok")
    } else {
        missing(CHECK, format!("no {}", facts.cxx_runtime_missing.join(", ")),
                "Microsoft's Visual C++ Redistributable (x64): `model-maker setup` offers to fetch it from \
                 Microsoft and start it (one administrator prompt)")
    }
}

fn path_row(facts: &Facts) -> Row {
    const CHECK: &str = "Folder path";
    let found = format!("{} characters", facts.repository_path_chars);
    match facts.long_paths {
        Some(true) => ok(CHECK, found, "ok (long paths are on)"),
        _ if facts.repository_path_chars <= WINDOWS_REPOSITORY_PATH_MOST => ok(CHECK, found, "ok"),
        _ => missing(CHECK, found, format!(
            "a folder whose path has at most {WINDOWS_REPOSITORY_PATH_MOST} characters (move it, e.g. under \
             C:\\model-maker), or long paths turned on")),
    }
}

fn python_row(python: &PythonFound) -> Row {
    let installs = format!("ok (setup installs its own {})", platform::PYTHON_VERSION);
    match python {
        PythonFound::Environment(version) => ok("Python", format!("{version} (.venv)"), "ok"),
        PythonFound::System(version) => ok("Python", format!("{version} (this PC's)"), installs),
        PythonFound::None => ok("Python", "none", installs),
    }
}

fn simplify_row(found: bool) -> Row {
    if found {
        ok("model-maker-simplify", "found", "ok")
    } else {
        ok(
            "model-maker-simplify",
            "not built",
            "setup builds it (needs Rust and a C++ compiler) or uses the one \
                                                 that came with this program",
        )
    }
}

fn git_words(system: System) -> &'static str {
    match system {
        System::Windows => {
            "git (for the view painter's own code: https://git-scm.com/download/win)"
        }
        System::MacOs => "git (xcode-select --install)",
        System::Linux => "git",
    }
}

fn tesseract_words(system: System) -> &'static str {
    match system {
        System::Linux => "reads pictures for painted lettering: tesseract 5 or newer with English data (Debian 12, \
                          Ubuntu 24.04: sudo apt install tesseract-ocr tesseract-ocr-eng; Ubuntu 22.04 ships 4: \
                          docs/setup.md)",
        System::Windows => "reads pictures for painted lettering: tesseract 5 or newer with English data \
                            (docs/setup.md, \"Windows\")",
        System::MacOs => "reads pictures for painted lettering: tesseract 5 or newer (brew install tesseract)",
    }
}

fn tesseract_row(tesseract: &Tesseract, system: System) -> Row {
    let words = tesseract_words(system);
    match tesseract {
        Tesseract::None => ok(
            "tesseract",
            "none",
            format!("optional, recommended: {words}"),
        ),
        Tesseract::Found { version: None, .. } => ok(
            "tesseract",
            "tesseract (version unknown)",
            format!("optional, not used (no version found): {words}"),
        ),
        Tesseract::Found {
            version: Some(v),
            english,
        } => {
            if !major(v).is_some_and(|m| m >= TESSERACT_MAJOR) {
                ok(
                    "tesseract",
                    format!("tesseract {v}"),
                    format!("optional, not used (too old): {words}"),
                )
            } else if !english {
                ok(
                    "tesseract",
                    format!("tesseract {v}, no English data"),
                    format!("optional, not used: {words}"),
                )
            } else {
                ok("tesseract", format!("tesseract {v}"), "ok")
            }
        }
    }
}

/// The name of the row that says whether setup ended well (the companion shows "ready" from it).
pub const SETUP_ROW: &str = "Setup";

fn setup_row(done: crate::setup::Done) -> Row {
    use crate::setup::Done;
    match done {
        Done::WithModels => ok(SETUP_ROW, "done", "ok"),
        Done::WithoutModels => ok(
            SETUP_ROW,
            "done, without the models",
            "`model-maker setup` fetches them",
        ),
        // Also a PC set up before setup kept this record: no proof of a good end is guessed from its folders.
        Done::No => ok(
            SETUP_ROW,
            "no record of a finished setup",
            "run `model-maker setup` (once more on a PC set up before: it carries on where it stopped)",
        ),
    }
}

fn judge_row(installed: bool) -> Row {
    if installed {
        ok(
            "question judge",
            format!("llama.cpp {}, Qwen 3.5 9B", platform::LLAMA_TAG),
            "ok",
        )
    } else {
        ok("question judge", "not installed", "optional: asks each finished object whether it is the thing named \
                                               (setup installs it; --without-question-judge leaves it out)")
    }
}

/// How many needed things are missing.
pub fn missing_count(rows: &[Row]) -> usize {
    rows.iter()
        .filter(|r| matches!(r.verdict, Verdict::Missing(_)))
        .count()
}

const FOUND_WIDTH: usize = 44;

/// The table as setup.sh prints it: check, found (cut to its column), result.
pub fn table(rows: &[Row]) -> String {
    let mut out = format!(
        "\n{:<22} {:<44} {}\n{:<22} {:<44} {}\n",
        "Check", "Found", "Result", "-----", "-----", "------"
    );
    for row in rows {
        let found: String = row.found.chars().take(FOUND_WIDTH).collect();
        let result = match &row.verdict {
            Verdict::Ok(words) => words.clone(),
            Verdict::Missing(needed) => format!("MISSING: {needed}"),
        };
        out.push_str(&format!("{:<22} {:<44} {}\n", row.check, found, result));
    }
    out
}

// --- reading what programs say (pure, tested) ---

/// `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`.
pub fn parse_cards(csv: &str) -> Vec<Card> {
    csv.lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(str::trim);
            let name = fields.next()?.to_string();
            let memory_mib = fields
                .next()?
                .chars()
                .filter(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok()?;
            let driver = fields.next().unwrap_or("").to_string();
            (!name.is_empty()).then_some(Card {
                name,
                memory_mib,
                driver,
            })
        })
        .collect()
}

/// The CUDA version in nvidia-smi's own table: "CUDA Version: 12.8", or "CUDA UMD Version: 13.5" in the header
/// newer drivers print (Windows 621.40: "NVIDIA-SMI 621.40  KMD Version: 621.40  CUDA UMD Version: 13.5").
pub fn parse_cuda(text: &str) -> Option<String> {
    ["CUDA Version:", "CUDA UMD Version:"]
        .iter()
        .filter_map(|label| text.split(label).nth(1))
        .map(|after| {
            after
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect::<String>()
        })
        .find(|version| !version.is_empty())
}

/// The first dotted number in a program's --version.
pub fn parse_version_number(text: &str) -> Option<String> {
    let line = text.lines().next()?;
    line.split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .map(|w| w.trim_matches('.'))
        .find(|w| w.contains('.') && w.split('.').all(|p| !p.is_empty()))
        .map(str::to_string)
}

/// `reg query HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled`: "... REG_DWORD 0x1".
pub fn parse_long_paths(text: &str) -> Option<bool> {
    let line = text.lines().find(|l| l.contains("LongPathsEnabled"))?;
    let value = line.split_whitespace().last()?;
    Some(value.trim_start_matches("0x") != "0")
}

// --- gathering from this PC ---

fn run(program: impl AsRef<std::ffi::OsStr>, args: &[&str]) -> Option<String> {
    let done = platform::quiet(Command::new(program).args(args))
        .output()
        .ok()?;
    let mut text = String::from_utf8_lossy(&done.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&done.stderr));
    done.status.success().then_some(text)
}

fn system_folder() -> PathBuf {
    PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into()))
        .join("System32")
}

/// nvidia-smi: by its full path under the system folder on Windows, never by name there.
pub fn nvidia_smi(system: System) -> PathBuf {
    if system == System::Windows {
        system_folder().join("nvidia-smi.exe")
    } else {
        PathBuf::from("nvidia-smi")
    }
}

fn gather_cards(system: System) -> (Cards, Option<String>) {
    let tool = nvidia_smi(system);
    let query = [
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ];
    match platform::quiet(Command::new(&tool).args(query)).output() {
        Err(_) => (Cards::NoTool, None),
        Ok(done) if !done.status.success() => (Cards::ToolFailed, None),
        Ok(done) => {
            let cards = parse_cards(&String::from_utf8_lossy(&done.stdout));
            let cuda = run(&tool, &[]).and_then(|t| parse_cuda(&t));
            if cards.is_empty() {
                (Cards::ToolFailed, None)
            } else {
                (Cards::Found(cards), cuda)
            }
        }
    }
}

fn folder_size_gb(folder: &Path) -> u64 {
    fn walk(folder: &Path) -> u64 {
        let Ok(entries) = std::fs::read_dir(folder) else {
            return 0;
        };
        entries
            .flatten()
            .map(|e| match e.file_type() {
                Ok(t) if t.is_dir() => walk(&e.path()),
                Ok(t) if t.is_file() => e.metadata().map(|m| m.len()).unwrap_or(0),
                _ => 0,
            })
            .sum()
    }
    walk(folder) / 1_000_000_000
}

fn free_disk_gb(folder: &Path) -> u64 {
    // Plain `C:\...`, so it compares with a disk's mount point (crate::paths).
    let real = crate::paths::real(folder);
    let Ok(disks) = sysinfo::Disks::new_with_refreshed_list() else {
        return 0;
    };
    disks
        .iter()
        .filter(|d| real.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space() / 1_000_000_000)
        .unwrap_or(0)
}

fn gather_python(repository: &Path, target: Option<&Target>) -> PythonFound {
    const VERSION: &str = "import platform; print(platform.python_version())";
    if let Some(target) = target {
        let venv = repository.join(".venv").join(target.venv_python);
        if let Some(v) = run(&venv, &["-I", "-c", VERSION]) {
            return PythonFound::Environment(v.trim().to_string());
        }
    }
    for name in ["python3", "python"] {
        if let Some(v) = run(name, &["-I", "-c", VERSION]) {
            return PythonFound::System(v.trim().to_string());
        }
    }
    PythonFound::None
}

fn gather_tesseract() -> Tesseract {
    let Ok(done) = platform::quiet(Command::new("tesseract").arg("--version")).output() else {
        return Tesseract::None;
    };
    let mut said = String::from_utf8_lossy(&done.stdout).into_owned();
    said.push_str(&String::from_utf8_lossy(&done.stderr));
    let english = run("tesseract", &["--list-langs"])
        .map(|t| t.lines().any(|l| l.trim() == "eng"))
        .unwrap_or(false);
    Tesseract::Found {
        version: parse_version_number(&said),
        english,
    }
}

/// The facts of this PC for a repository at `repository`.
pub fn gather(repository: &Path, models: &Path) -> Facts {
    let (system, processor) = platform::here();
    let target = platform::target(system, processor);
    let (cards, cuda) = gather_cards(system);
    let memory_gib = sysinfo::System::new()
        .map(|mut memory| {
            memory.refresh_memory();
            memory.total_memory() / (1 << 30)
        })
        .unwrap_or(0);
    let tools = repository.join(".tools");
    let judge_server = tools
        .join(format!("llama-{}", platform::LLAMA_TAG))
        .join(platform::program(system, "llama-server"));
    let judge_model = models.join("qwen3.5-9b-q4");
    Facts {
        system,
        processor,
        wsl: system == System::Linux
            && std::fs::read_to_string("/proc/version")
                .is_ok_and(|v| v.to_lowercase().contains("microsoft")),
        cards,
        cuda,
        memory_gib,
        free_disk_gb: free_disk_gb(repository),
        models_gb: folder_size_gb(models),
        python: gather_python(repository, target),
        git: run("git", &["--version"])
            .and_then(|t| parse_version_number(&t))
            .map(|v| format!("git {v}")),
        tesseract: gather_tesseract(),
        setup_done: crate::setup::done_record(&tools),
        judge_installed: judge_server.is_file()
            && judge_model.join("Qwen_Qwen3.5-9B-Q4_K_M.gguf").is_file()
            && judge_model
                .join("mmproj-Qwen_Qwen3.5-9B-f16.gguf")
                .is_file(),
        simplify_found: crate::tools::simplify_tool(
            repository,
            system,
            crate::tools::program_folder().as_deref(),
        )
        .is_some(),
        cxx_runtime_missing: if system == System::Windows {
            let folder = system_folder();
            WINDOWS_CXX_RUNTIME
                .iter()
                .copied()
                .filter(|f| !folder.join(f).is_file())
                .collect()
        } else {
            Vec::new()
        },
        long_paths: (system == System::Windows)
            .then(|| {
                run(
                    system_folder().join("reg.exe"),
                    &[
                        "query",
                        r"HKLM\SYSTEM\CurrentControlSet\Control\FileSystem",
                        "/v",
                        "LongPathsEnabled",
                    ],
                )
            })
            .flatten()
            .and_then(|t| parse_long_paths(&t)),
        repository_path_chars: repository.as_os_str().len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{chosen_lock, target, Tier};

    fn linux_pc() -> Facts {
        Facts {
            system: System::Linux,
            processor: Processor::X86_64,
            wsl: false,
            cards: Cards::Found(vec![Card {
                name: "NVIDIA GeForce RTX 3060".into(),
                memory_mib: 12288,
                driver: "580.95".into(),
            }]),
            cuda: Some("12.8".into()),
            memory_gib: 45,
            free_disk_gb: 900,
            models_gb: 0,
            python: PythonFound::None,
            git: Some("git 2.47.3".into()),
            tesseract: Tesseract::Found {
                version: Some("5.3.0".into()),
                english: true,
            },
            judge_installed: false,
            setup_done: crate::setup::Done::No,
            simplify_found: false,
            cxx_runtime_missing: Vec::new(),
            long_paths: None,
            repository_path_chars: 30,
        }
    }

    fn fresh_windows() -> Facts {
        Facts {
            system: System::Windows,
            cards: Cards::NoTool,
            cuda: None,
            memory_gib: 7,
            free_disk_gb: 49,
            git: None,
            tesseract: Tesseract::None,
            cxx_runtime_missing: WINDOWS_CXX_RUNTIME.to_vec(),
            long_paths: Some(false),
            repository_path_chars: 52,
            ..linux_pc()
        }
    }

    fn judged(facts: &Facts) -> Vec<Row> {
        let t = target(facts.system, facts.processor);
        let lock = t.and_then(|t| chosen_lock(t, None).ok());
        rows(facts, t, lock, true)
    }

    fn verdict<'a>(rows: &'a [Row], check: &str) -> &'a Verdict {
        &rows
            .iter()
            .find(|r| r.check == check)
            .unwrap_or_else(|| panic!("no row {check}"))
            .verdict
    }

    #[test]
    fn a_measured_linux_pc_with_a_12_gb_card_has_nothing_missing() {
        let rows = judged(&linux_pc());
        assert_eq!(missing_count(&rows), 0, "{}", table(&rows));
        assert!(
            matches!(verdict(&rows, "Graphics card"), Verdict::Ok(w) if w.contains("take turns"))
        );
        assert!(table(&rows).contains(Tier::Measured.words()));
        assert!(table(&rows).contains("unsigned build"));
    }

    #[test]
    fn a_fresh_windows_pc_says_each_thing_it_lacks() {
        let rows = judged(&fresh_windows());
        for check in [
            "Graphics card",
            "Memory",
            "Free disk here",
            "C++ runtime",
            "git",
        ] {
            assert!(
                matches!(verdict(&rows, check), Verdict::Missing(_)),
                "{check}: {}",
                table(&rows)
            );
        }
        assert!(matches!(verdict(&rows, "Folder path"), Verdict::Ok(_)));
        assert!(
            matches!(verdict(&rows, "C++ runtime"), Verdict::Missing(w) if w.contains("administrator"))
        );
        assert!(matches!(verdict(&rows, "System"), Verdict::Ok(w) if w == "ok"));
        assert!(table(&rows).contains(Tier::Supported.words()));
        assert!(!table(&rows).contains("experimental"));
    }

    #[test]
    fn a_long_windows_path_is_refused_unless_long_paths_are_on() {
        let mut facts = fresh_windows();
        facts.repository_path_chars = WINDOWS_REPOSITORY_PATH_MOST + 1;
        assert!(matches!(
            verdict(&judged(&facts), "Folder path"),
            Verdict::Missing(_)
        ));
        facts.long_paths = Some(true);
        assert!(matches!(
            verdict(&judged(&facts), "Folder path"),
            Verdict::Ok(_)
        ));
        facts.repository_path_chars = WINDOWS_REPOSITORY_PATH_MOST;
        facts.long_paths = None;
        assert!(matches!(
            verdict(&judged(&facts), "Folder path"),
            Verdict::Ok(_)
        ));
    }

    #[test]
    fn windows_only_rows_are_not_shown_elsewhere() {
        let rows = judged(&linux_pc());
        assert!(!rows
            .iter()
            .any(|r| r.check == "C++ runtime" || r.check == "Folder path"));
    }

    #[test]
    fn cards_and_cuda() {
        let mut facts = linux_pc();
        facts.cards = Cards::Found(vec![
            Card {
                name: "small".into(),
                memory_mib: 8192,
                driver: "1".into(),
            },
            Card {
                name: "NVIDIA GeForce RTX 3080 Laptop GPU".into(),
                memory_mib: 16384,
                driver: "1".into(),
            },
        ]);
        let rows = judged(&facts);
        assert_eq!(verdict(&rows, "Graphics card"), &Verdict::Ok("ok".into()));
        facts.cards = Cards::Found(vec![Card {
            name: "small".into(),
            memory_mib: 8192,
            driver: "1".into(),
        }]);
        assert!(matches!(
            verdict(&judged(&facts), "Graphics card"),
            Verdict::Missing(_)
        ));
        facts.cards = Cards::ToolFailed;
        assert!(
            matches!(verdict(&judged(&facts), "Graphics card"), Verdict::Missing(w) if w.contains("driver"))
        );
        facts = linux_pc();
        facts.cuda = Some("11.8".into());
        assert!(matches!(
            verdict(&judged(&facts), "CUDA"),
            Verdict::Missing(_)
        ));
        facts.cuda = None;
        assert!(matches!(
            verdict(&judged(&facts), "CUDA"),
            Verdict::Missing(_)
        ));
    }

    #[test]
    fn a_processor_lock_needs_no_card() {
        let mut facts = linux_pc();
        facts.cards = Cards::NoTool;
        let t = target(System::Linux, Processor::X86_64).unwrap();
        let lock = chosen_lock(t, Some("linux-x86_64-cpu")).unwrap();
        let rows = rows(&facts, Some(t), Some(lock), false);
        assert_eq!(missing_count(&rows), 0, "{}", table(&rows));
        assert!(!rows.iter().any(|r| r.check == "CUDA"));
    }

    #[test]
    fn a_mac_is_not_supported_yet_and_not_missing_a_card() {
        let mut facts = linux_pc();
        facts.system = System::MacOs;
        facts.processor = Processor::Aarch64;
        facts.cards = Cards::NoTool;
        let rows = judged(&facts);
        assert_eq!(missing_count(&rows), 0, "{}", table(&rows));
        assert!(table(&rows).contains("shared"));
        assert!(table(&rows).contains(Tier::NotSupportedYet.words()));
        assert!(!table(&rows).contains("experimental"));
    }

    #[test]
    fn linux_arm64_says_it_has_no_lock() {
        let mut facts = linux_pc();
        facts.processor = Processor::Aarch64;
        assert!(
            matches!(verdict(&judged(&facts), "Processor"), Verdict::Missing(w) if w.contains("arm64"))
        );
    }

    #[test]
    fn memory_and_disk_limits() {
        let mut facts = linux_pc();
        facts.memory_gib = 16;
        assert!(
            matches!(verdict(&judged(&facts), "Memory"), Verdict::Ok(w) if w.starts_with("try it"))
        );
        // linux_pc's card is 12 GB: 24 GB is enough only with a 16 GB card (setup.sh's check_memory).
        facts.memory_gib = MEMORY_GOOD_GIB;
        assert!(
            matches!(verdict(&judged(&facts), "Memory"), Verdict::Ok(w) if w.starts_with("try it (32 GB"))
        );
        facts.memory_gib = MEMORY_GOOD_GIB_SMALL_CARD;
        assert!(matches!(verdict(&judged(&facts), "Memory"), Verdict::Ok(w) if w == "ok"));
        facts.cards = Cards::Found(vec![Card {
            name: "NVIDIA GeForce RTX 3080 Laptop GPU".into(),
            memory_mib: 16384,
            driver: "580.95".into(),
        }]);
        facts.memory_gib = MEMORY_GOOD_GIB;
        assert!(matches!(verdict(&judged(&facts), "Memory"), Verdict::Ok(w) if w == "ok"));
        facts.memory_gib = 15;
        assert!(matches!(
            verdict(&judged(&facts), "Memory"),
            Verdict::Missing(_)
        ));
        assert_eq!(disk_needed_gb(System::Linux, 0), 77);
        assert_eq!(disk_needed_gb(System::Windows, 0), 82);
        assert_eq!(disk_needed_gb(System::Linux, 57), 20);
        assert_eq!(disk_needed_gb(System::Linux, 200), DISK_LEAST_GB);
    }

    #[test]
    fn windows_verbatim_paths_compare_with_disks() {
        assert_eq!(
            crate::paths::simplified(PathBuf::from(r"\\?\C:\mm-launcher-test\repo")),
            PathBuf::from(r"C:\mm-launcher-test\repo")
        );
        assert_eq!(
            crate::paths::simplified(PathBuf::from(r"\\?\UNC\server\share")),
            PathBuf::from(r"\\?\UNC\server\share"),
            "a network path is left as it is"
        );
        assert_eq!(
            crate::paths::simplified(PathBuf::from("/srv/repository")),
            PathBuf::from("/srv/repository")
        );
    }

    #[test]
    fn tests_on_the_processor_need_no_makes_memory() {
        let mut facts = fresh_windows();
        facts.memory_gib = 7;
        let t = target(System::Windows, Processor::X86_64);
        let lock = t.and_then(|t| chosen_lock(t, None).ok());
        let for_tests = rows(&facts, t, lock, false);
        assert!(matches!(verdict(&for_tests, "Memory"), Verdict::Ok(w) if w.contains("tests")));
        assert!(matches!(
            verdict(&judged(&facts), "Memory"),
            Verdict::Missing(_)
        ));
    }

    #[test]
    fn tesseract_versions() {
        let rows_for = |t: Tesseract| {
            let mut facts = linux_pc();
            facts.tesseract = t;
            judged(&facts)
        };
        let words = |rows: &[Row]| match verdict(rows, "tesseract") {
            Verdict::Ok(w) => w.clone(),
            m => panic!("{m:?}"),
        };
        assert_eq!(
            words(&rows_for(Tesseract::Found {
                version: Some("5.5.0".into()),
                english: true
            })),
            "ok"
        );
        assert!(words(&rows_for(Tesseract::Found {
            version: Some("4.1.1".into()),
            english: true
        }))
        .contains("too old"));
        assert!(words(&rows_for(Tesseract::Found {
            version: Some("5.5.0".into()),
            english: false
        }))
        .contains("not used"));
        assert!(words(&rows_for(Tesseract::None)).starts_with("optional"));
    }

    #[test]
    fn the_table_cuts_the_found_column_and_marks_missing() {
        let rows = vec![missing("git", "x".repeat(60), "git")];
        let printed = table(&rows);
        assert!(printed.contains(&format!("{} MISSING: git", "x".repeat(44))));
        assert!(!printed.contains(&"x".repeat(45)));
    }

    #[test]
    fn reading_nvidia_smi() {
        let cards = parse_cards("NVIDIA GeForce RTX 3060, 12288, 580.95.05\nNVIDIA GeForce RTX 3080 Laptop GPU, 16384, 621.40\n");
        assert_eq!(cards.len(), 2);
        assert_eq!(
            cards[1],
            Card {
                name: "NVIDIA GeForce RTX 3080 Laptop GPU".into(),
                memory_mib: 16384,
                driver: "621.40".into()
            }
        );
        assert_eq!(
            parse_cards("NVIDIA GeForce RTX 3060, [N/A], 621.40\n"),
            vec![]
        );
        assert_eq!(parse_cards(""), vec![]);
        let table = "| NVIDIA-SMI 621.40      Driver Version: 621.40      CUDA Version: 13.1     |";
        assert_eq!(parse_cuda(table).as_deref(), Some("13.1"));
        // The header NVIDIA's Windows driver 621.40 printed in the Windows test of 26 November 2026.
        let umd = "NVIDIA-SMI 621.40 KMD Version: 621.40  CUDA UMD Version: 13.5";
        assert_eq!(parse_cuda(umd).as_deref(), Some("13.5"));
        assert_eq!(parse_cuda("no such line"), None);
    }

    #[test]
    fn reading_versions_and_the_registry() {
        assert_eq!(
            parse_version_number("git version 2.56.1.windows.1").as_deref(),
            Some("2.56.1")
        );
        assert_eq!(
            parse_version_number("tesseract 5.3.0\n leptonica-1.82.0").as_deref(),
            Some("5.3.0")
        );
        assert_eq!(parse_version_number("tesseract (no number)"), None);
        let reg = "\r\nHKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\r\n    LongPathsEnabled    REG_DWORD    0x0\r\n";
        assert_eq!(parse_long_paths(reg), Some(false));
        assert_eq!(parse_long_paths(&reg.replace("0x0", "0x1")), Some(true));
        assert_eq!(parse_long_paths("ERROR: not found"), None);
    }

    #[test]
    fn nvidia_smi_by_full_path_on_windows() {
        assert!(
            nvidia_smi(System::Windows).ends_with("System32/nvidia-smi.exe")
                || nvidia_smi(System::Windows)
                    .to_string_lossy()
                    .ends_with("System32\\nvidia-smi.exe")
        );
        assert_eq!(nvidia_smi(System::Linux), PathBuf::from("nvidia-smi"));
    }

    /// setup.sh's `readonly NAME=VALUE` lines.
    fn setup_sh_value(name: &str) -> String {
        let setup = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../setup.sh");
        let text = std::fs::read_to_string(setup).unwrap();
        let prefix = format!("readonly {name}=");
        let line = text
            .lines()
            .find(|l| l.starts_with(&prefix))
            .unwrap_or_else(|| panic!("no {name} in setup.sh"));
        line[prefix.len()..]
            .split_whitespace()
            .next()
            .unwrap()
            .trim_matches('"')
            .to_string()
    }

    #[test]
    fn limits_and_pins_equal_setup_sh() {
        assert_eq!(setup_sh_value("GPU_GOOD_MIB"), GPU_GOOD_MIB.to_string());
        assert_eq!(setup_sh_value("GPU_TRY_MIB"), GPU_TRY_MIB.to_string());
        assert_eq!(setup_sh_value("CUDA_MAJOR"), CUDA_MAJOR.to_string());
        assert_eq!(
            setup_sh_value("MEMORY_GOOD_GIB"),
            MEMORY_GOOD_GIB.to_string()
        );
        assert_eq!(
            setup_sh_value("MEMORY_GOOD_GIB_SMALL_CARD"),
            MEMORY_GOOD_GIB_SMALL_CARD.to_string()
        );
        assert_eq!(
            setup_sh_value("MEMORY_LEAST_GIB"),
            MEMORY_LEAST_GIB.to_string()
        );
        assert_eq!(setup_sh_value("DISK_GB"), DISK_GB.to_string());
        assert_eq!(setup_sh_value("PYTHON_VERSION"), platform::PYTHON_VERSION);
        assert_eq!(setup_sh_value("UV_VERSION"), platform::UV_VERSION);
        assert_eq!(setup_sh_value("LLAMA_TAG"), platform::LLAMA_TAG);
        let linux = target(System::Linux, Processor::X86_64).unwrap();
        let arm = target(System::Linux, Processor::Aarch64).unwrap();
        assert_eq!(setup_sh_value("UV_SHA256_X86_64"), linux.uv.sha256);
        assert_eq!(setup_sh_value("UV_SHA256_AARCH64"), arm.uv.sha256);
        assert_eq!(
            setup_sh_value("LLAMA_SHA256_X86_64"),
            linux.llama.unwrap().sha256
        );
    }

    #[test]
    fn gathering_this_pc_does_not_fail() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let facts = gather(&repository, &repository.join("models"));
        assert_eq!((facts.system, facts.processor), platform::here());
        assert!(facts.memory_gib > 0);
    }
}
