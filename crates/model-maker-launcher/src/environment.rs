//! The environment each child program gets: uv's (exactly setup.sh's settings, nothing inherited that could point
//! it at another index) and the Python side's (exactly the bash `model-maker` command's, with one difference on
//! Windows).

use std::ffi::{OsStr, OsString};
use std::path::Path;

use crate::platform::System;

/// A child's environment: what to remove from the inherited one, and what to set.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Changes {
    pub removed: Vec<OsString>,
    pub set: Vec<(OsString, OsString)>,
}

impl Changes {
    pub fn apply(&self, command: &mut std::process::Command) {
        for name in &self.removed {
            command.env_remove(name);
        }
        for (name, value) in &self.set {
            command.env(name, value);
        }
    }

    pub fn value(&self, name: &str) -> Option<&OsStr> {
        self.set
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_os_str())
    }
}

/// uv's settings: its cache and its Pythons inside the repository's `.tools`, no configuration file, only Pythons it
/// installed; every other UV_* and PIP_* variable removed (an index, a mirror or a constraint from the user's own
/// setup must not change what is installed).
pub fn for_uv(tools: &Path, inherited: impl IntoIterator<Item = (OsString, OsString)>) -> Changes {
    let removed = inherited
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| {
            let upper = name.to_string_lossy().to_ascii_uppercase();
            upper.starts_with("UV_") || upper.starts_with("PIP_")
        })
        .collect();
    let set = vec![
        (
            "UV_CACHE_DIR".into(),
            tools.join("uv-cache").into_os_string(),
        ),
        (
            "UV_PYTHON_INSTALL_DIR".into(),
            tools.join("python").into_os_string(),
        ),
        ("UV_NO_CONFIG".into(), "1".into()),
        ("UV_PYTHON_PREFERENCE".into(), "only-managed".into()),
    ];
    Changes { removed, set }
}

/// The Python side's environment, as the bash `model-maker` sets it: the package on PYTHONPATH (before what was
/// there), the models and their code, nothing fetched at run time, the card's memory in growable pieces (Linux
/// only: not supported on Windows). On Windows also PYTHONUTF8=1: without it Python writes to a pipe or a file in
/// the console's code page and fails on a character outside it (the Windows test, 25 Nov 2026).
pub fn for_python(
    system: System,
    repository: &Path,
    models: &Path,
    sources: &Path,
    inherited: impl IntoIterator<Item = (OsString, OsString)>,
) -> Changes {
    let inherited: Vec<(OsString, OsString)> = inherited.into_iter().collect();
    let old = |name: &str| {
        inherited
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
    };
    let mut python_path: Vec<std::path::PathBuf> = vec![repository.join("python")];
    if let Some(before) = old("PYTHONPATH").filter(|v| !v.is_empty()) {
        python_path.extend(std::env::split_paths(&before));
    }
    let joined = std::env::join_paths(&python_path)
        .unwrap_or_else(|_| repository.join("python").into_os_string());
    let mut set: Vec<(OsString, OsString)> = vec![
        ("PYTHONPATH".into(), joined),
        (
            "MODEL_MAKER_MODELS".into(),
            old("MODEL_MAKER_MODELS").unwrap_or_else(|| models.into()),
        ),
        (
            "MODEL_MAKER_SOURCES".into(),
            old("MODEL_MAKER_SOURCES").unwrap_or_else(|| sources.into()),
        ),
        ("HF_HUB_OFFLINE".into(), "1".into()),
        ("TRANSFORMERS_OFFLINE".into(), "1".into()),
    ];
    if system == System::Linux {
        let alloc =
            old("PYTORCH_CUDA_ALLOC_CONF").unwrap_or_else(|| "expandable_segments:True".into());
        set.push(("PYTORCH_CUDA_ALLOC_CONF".into(), alloc));
    }
    if system == System::Windows {
        set.push(("PYTHONUTF8".into(), "1".into()));
    }
    Changes {
        removed: Vec::new(),
        set,
    }
}

/// MODEL_MAKER_SIMPLIFY pointing at the tool that came with this program (simplify.py finds it there before
/// anything else), unless the person set it.
pub fn simplify_setting(
    tool: Option<&Path>,
    inherited: &[(OsString, OsString)],
) -> Option<(OsString, OsString)> {
    const NAME: &str = "MODEL_MAKER_SIMPLIFY";
    if inherited.iter().any(|(n, _)| n == NAME) {
        return None;
    }
    tool.map(|tool| (NAME.into(), tool.as_os_str().to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn pairs(list: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
        list.iter()
            .map(|(n, v)| (OsString::from(n), OsString::from(v)))
            .collect()
    }

    #[test]
    fn uv_gets_setup_sh_settings_and_nothing_inherited_of_its_own() {
        let tools = PathBuf::from("/r/.tools");
        let changes = for_uv(
            &tools,
            pairs(&[
                ("UV_INDEX_URL", "https://mirror"),
                ("PIP_INDEX_URL", "x"),
                ("pip_extra_index_url", "y"),
                ("PATH", "/bin"),
                ("HOME", "/h"),
            ]),
        );
        assert_eq!(
            changes.removed,
            vec![
                OsString::from("UV_INDEX_URL"),
                "PIP_INDEX_URL".into(),
                "pip_extra_index_url".into()
            ]
        );
        assert_eq!(
            changes.value("UV_CACHE_DIR"),
            Some(tools.join("uv-cache").as_os_str())
        );
        assert_eq!(
            changes.value("UV_PYTHON_INSTALL_DIR"),
            Some(tools.join("python").as_os_str())
        );
        assert_eq!(changes.value("UV_NO_CONFIG"), Some(OsStr::new("1")));
        assert_eq!(
            changes.value("UV_PYTHON_PREFERENCE"),
            Some(OsStr::new("only-managed"))
        );
        assert!(
            changes.value("HF_TOKEN").is_none(),
            "nothing about the hub's token reaches uv"
        );
    }

    #[test]
    fn python_gets_the_bash_commands_environment() {
        let repository = PathBuf::from("/r");
        let changes = for_python(
            System::Linux,
            &repository,
            &repository.join("models"),
            &repository.join("sources"),
            pairs(&[("PYTHONPATH", "/extra"), ("PATH", "/bin")]),
        );
        let expected_path =
            std::env::join_paths([repository.join("python"), PathBuf::from("/extra")]).unwrap();
        assert_eq!(changes.value("PYTHONPATH"), Some(expected_path.as_os_str()));
        // PathBuf::join writes `\` between the parts on Windows: the expected value is built the same way.
        assert_eq!(
            changes.value("MODEL_MAKER_MODELS"),
            Some(repository.join("models").as_os_str())
        );
        assert_eq!(changes.value("HF_HUB_OFFLINE"), Some(OsStr::new("1")));
        assert_eq!(changes.value("TRANSFORMERS_OFFLINE"), Some(OsStr::new("1")));
        assert_eq!(
            changes.value("PYTORCH_CUDA_ALLOC_CONF"),
            Some(OsStr::new("expandable_segments:True"))
        );
        assert!(changes.value("PYTHONUTF8").is_none());
    }

    #[test]
    fn the_users_own_folders_and_setting_win() {
        let repository = PathBuf::from("/r");
        let changes = for_python(
            System::Linux,
            &repository,
            &repository.join("models"),
            &repository.join("sources"),
            pairs(&[
                ("MODEL_MAKER_MODELS", "/big/models"),
                ("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512"),
            ]),
        );
        assert_eq!(
            changes.value("MODEL_MAKER_MODELS"),
            Some(OsStr::new("/big/models"))
        );
        assert_eq!(
            changes.value("PYTORCH_CUDA_ALLOC_CONF"),
            Some(OsStr::new("max_split_size_mb:512"))
        );
    }

    #[test]
    fn the_shipped_simplify_tool_unless_the_person_named_one() {
        let tool = PathBuf::from("/r/bin/model-maker-simplify");
        assert_eq!(
            simplify_setting(Some(&tool), &[]),
            Some(("MODEL_MAKER_SIMPLIFY".into(), tool.clone().into_os_string()))
        );
        assert_eq!(
            simplify_setting(Some(&tool), &pairs(&[("MODEL_MAKER_SIMPLIFY", "/mine")])),
            None
        );
        assert_eq!(simplify_setting(None, &[]), None);
    }

    #[test]
    fn windows_differences() {
        let repository = PathBuf::from("C:/r");
        let changes = for_python(
            System::Windows,
            &repository,
            &repository.join("models"),
            &repository.join("sources"),
            Vec::new(),
        );
        assert_eq!(changes.value("PYTHONUTF8"), Some(OsStr::new("1")));
        assert!(
            changes.value("PYTORCH_CUDA_ALLOC_CONF").is_none(),
            "not supported on Windows"
        );
        let macos = for_python(
            System::MacOs,
            &repository,
            &repository.join("m"),
            &repository.join("s"),
            Vec::new(),
        );
        assert!(
            macos.value("PYTORCH_CUDA_ALLOC_CONF").is_none() && macos.value("PYTHONUTF8").is_none()
        );
    }
}
