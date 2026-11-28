//! Pinned downloads: fetched from their makers' own addresses over TLS (checked by the operating system's own
//! verifier, so a company proxy's certificate works), written to `.part`, refused unless the sha256 is the pinned
//! one, then renamed. A failed or refused download is deleted and starts again from nothing next time (the launcher
//! fetches only small files; the big ones come through uv and Python). Archives are unpacked only after every
//! entry's path is checked to stay inside the folder.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::platform::{Archive, Download};

#[derive(Debug)]
pub enum Failed {
    Network(String, String),
    WrongSum {
        url: String,
        found: String,
        pinned: String,
    },
    Io(String, io::Error),
    Unsafe(String),
}

impl std::fmt::Display for Failed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failed::Network(url, why) => write!(f, "{url}: {why}"),
            Failed::WrongSum { url, found, pinned } => write!(
                f,
                "{url}: its sha256 is {found}, not the pinned {pinned}; refused and deleted"
            ),
            Failed::Io(what, error) => write!(f, "{what}: {error}"),
            Failed::Unsafe(why) => write!(f, "refused to unpack: {why}"),
        }
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The sha256 of a file, in lower-case hex.
pub fn sha256_of(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    io::copy(&mut file, &mut DigestWriter(&mut digest))?;
    Ok(hex(&digest.finalize()))
}

struct DigestWriter<'a>(&'a mut Sha256);

impl Write for DigestWriter<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.0.update(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn part_of(into: &Path) -> PathBuf {
    let mut name = into.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    into.with_file_name(name)
}

/// What a reader gives, written to `into` only if its sha256 is `pinned`; else nothing is left behind.
pub fn written_if_pinned(
    mut source: impl Read,
    url: &str,
    pinned: &str,
    into: &Path,
) -> Result<(), Failed> {
    let part = part_of(into);
    let io_failed = |e| Failed::Io(part.display().to_string(), e);
    let mut file = File::create(&part).map_err(io_failed)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    let copied = loop {
        match source.read(&mut buffer) {
            Ok(0) => break Ok(()),
            Ok(n) => {
                digest.update(&buffer[..n]);
                if let Err(e) = file.write_all(&buffer[..n]) {
                    break Err(Failed::Io(part.display().to_string(), e));
                }
            }
            Err(e) => break Err(Failed::Network(url.to_string(), e.to_string())),
        }
    };
    drop(file);
    let found = hex(&digest.finalize());
    if let Err(failed) = copied {
        let _ = std::fs::remove_file(&part);
        return Err(failed);
    }
    if found != pinned {
        let _ = std::fs::remove_file(&part);
        return Err(Failed::WrongSum {
            url: url.to_string(),
            found,
            pinned: pinned.to_string(),
        });
    }
    std::fs::rename(&part, into).map_err(|e| Failed::Io(into.display().to_string(), e))
}

/// A stalled connection must not hang setup for ever (ureq's own limits are all "none"): connecting, the answer's
/// headers, and every single read of the body each get a limit; a slow but moving download is never cut off.
const CONNECT_LIMIT: Duration = Duration::from_secs(30);
const RESPONSE_LIMIT: Duration = Duration::from_secs(60);
const READ_LIMIT: Duration = Duration::from_secs(60);

fn agent() -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder()
        .root_certs(ureq::tls::RootCerts::PlatformVerifier)
        .unversioned_rustls_crypto_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .build();
    ureq::Agent::config_builder()
        .tls_config(tls)
        .https_only(true)
        .timeout_connect(Some(CONNECT_LIMIT))
        .timeout_recv_response(Some(RESPONSE_LIMIT))
        .timeout_per_read(Some(READ_LIMIT))
        .build()
        .new_agent()
}

/// One try, and one more after a failure of the network or the disk (never after a wrong sum).
fn tried_twice(mut attempt: impl FnMut() -> Result<(), Failed>) -> Result<(), Failed> {
    match attempt() {
        Err(failed @ (Failed::Network(..) | Failed::Io(..))) => {
            eprintln!("{failed}; trying once more");
            attempt()
        }
        done => done,
    }
}

/// A pinned file fetched into `into` (its sum checked before it is kept).
pub fn fetched(download: &Download, into: &Path) -> Result<(), Failed> {
    tried_twice(|| {
        let response = agent()
            .get(download.url)
            .call()
            .map_err(|e| Failed::Network(download.url.to_string(), e.to_string()))?;
        let reader = response.into_body().into_reader();
        written_if_pinned(reader, download.url, download.sha256, into)
    })
}

/// A file whose sum cannot be pinned (its maker replaces it with each update), fetched into `into`: the caller must
/// check it another way before using it (the C++ runtime's installer: its signature).
pub fn fetched_unpinned(url: &str, into: &Path) -> Result<(), Failed> {
    tried_twice(|| {
        let response = agent()
            .get(url)
            .call()
            .map_err(|e| Failed::Network(url.to_string(), e.to_string()))?;
        let mut reader = response.into_body().into_reader();
        let part = part_of(into);
        let written = File::create(&part)
            .and_then(|mut file| io::copy(&mut reader, &mut file))
            .and_then(|_| std::fs::rename(&part, into));
        written.map_err(|e| {
            let _ = std::fs::remove_file(&part);
            Failed::Io(url.to_string(), e)
        })
    })
}

/// An archive entry's path with `strip` leading parts removed; None for the entry that is only a stripped folder.
/// Refused when it is absolute, has a drive or a `..`, so nothing lands outside the folder.
pub fn inside_path(entry: &Path, strip: usize) -> Result<Option<PathBuf>, Failed> {
    let mut kept = PathBuf::new();
    let mut seen = 0;
    for component in entry.components() {
        match component {
            Component::Normal(part) => {
                seen += 1;
                if seen > strip {
                    kept.push(part);
                }
            }
            Component::CurDir => {}
            _ => {
                return Err(Failed::Unsafe(format!(
                    "{} leaves the folder",
                    entry.display()
                )))
            }
        }
    }
    Ok((!kept.as_os_str().is_empty()).then_some(kept))
}

/// A link's target must stay beside it or below it.
fn safe_link_target(target: &Path) -> bool {
    target
        .components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

fn unpacked_tar_gz(archive: &Path, into: &Path, strip: usize) -> Result<(), Failed> {
    let io_failed = |e| Failed::Io(archive.display().to_string(), e);
    let file = File::open(archive).map_err(io_failed)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries().map_err(io_failed)? {
        let mut entry = entry.map_err(io_failed)?;
        let path = entry.path().map_err(io_failed)?.into_owned();
        let Some(relative) = inside_path(&path, strip)? else {
            continue;
        };
        let kind = entry.header().entry_type();
        if kind.is_symlink() || kind.is_hard_link() {
            let target = entry.link_name().map_err(io_failed)?.unwrap_or_default();
            if !safe_link_target(&target) || kind.is_hard_link() {
                return Err(Failed::Unsafe(format!(
                    "{} links to {}",
                    path.display(),
                    target.display()
                )));
            }
        } else if !(kind.is_file() || kind.is_dir()) {
            return Err(Failed::Unsafe(format!(
                "{} is neither a file nor a folder",
                path.display()
            )));
        }
        let destination = into.join(&relative);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).map_err(io_failed)?;
        }
        entry.unpack(&destination).map_err(io_failed)?;
    }
    Ok(())
}

fn unpacked_zip(archive: &Path, into: &Path, strip: usize) -> Result<(), Failed> {
    let io_failed = |e| Failed::Io(archive.display().to_string(), e);
    let file = File::open(archive).map_err(io_failed)?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| Failed::Io(archive.display().to_string(), e.into()))?;
    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .map_err(|e| Failed::Io(archive.display().to_string(), e.into()))?;
        let name = entry
            .name()
            .map_err(|e| Failed::Io(archive.display().to_string(), e.into()))?;
        let name = PathBuf::from(name.replace('\\', "/"));
        if entry.is_symlink() {
            return Err(Failed::Unsafe(format!("{} is a link", name.display())));
        }
        let Some(relative) = inside_path(&name, strip)? else {
            continue;
        };
        let destination = into.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&destination).map_err(io_failed)?;
            continue;
        }
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).map_err(io_failed)?;
        }
        let mut out = File::create(&destination).map_err(io_failed)?;
        io::copy(&mut entry, &mut out).map_err(io_failed)?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&destination, std::fs::Permissions::from_mode(mode & 0o755))
                .map_err(io_failed)?;
        }
    }
    Ok(())
}

/// An archive unpacked into `into`, its first `strip` path parts removed (as tar's --strip-components).
pub fn unpacked(archive: &Path, kind: Archive, into: &Path, strip: usize) -> Result<(), Failed> {
    std::fs::create_dir_all(into).map_err(|e| Failed::Io(into.display().to_string(), e))?;
    match kind {
        Archive::TarGz => unpacked_tar_gz(archive, into, strip),
        Archive::Zip => unpacked_zip(archive, into, strip),
        Archive::Plain => Err(Failed::Unsafe(format!(
            "{} is not an archive",
            archive.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "mm-launcher-download-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    const HELLO_SHA256: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn a_pinned_file_is_kept_and_a_wrong_one_leaves_nothing() {
        let folder = temporary("pinned");
        let into = folder.join("uv.tar.gz");
        written_if_pinned(&b"hello"[..], "https://example.org/a", HELLO_SHA256, &into).unwrap();
        assert_eq!(std::fs::read(&into).unwrap(), b"hello");
        assert_eq!(sha256_of(&into).unwrap(), HELLO_SHA256);
        let other = folder.join("other.zip");
        let refused =
            written_if_pinned(&b"hellO"[..], "https://example.org/b", HELLO_SHA256, &other)
                .unwrap_err();
        assert!(matches!(refused, Failed::WrongSum { .. }), "{refused}");
        assert!(refused.to_string().contains("refused and deleted"));
        assert!(!other.exists() && !part_of(&other).exists());
        std::fs::remove_dir_all(&folder).unwrap();
    }

    struct BrokenReader;
    impl Read for BrokenReader {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::ConnectionReset, "reset"))
        }
    }

    #[test]
    fn a_broken_download_leaves_nothing() {
        let folder = temporary("broken");
        let into = folder.join("x");
        assert!(matches!(
            written_if_pinned(BrokenReader, "u", HELLO_SHA256, &into),
            Err(Failed::Network(..))
        ));
        assert!(!into.exists() && !part_of(&into).exists());
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn downloads_have_time_limits_and_one_retry() {
        let timeouts = agent().config().timeouts();
        assert_eq!(timeouts.connect, Some(CONNECT_LIMIT));
        assert_eq!(timeouts.recv_response, Some(RESPONSE_LIMIT));
        assert_eq!(agent().config().timeout_per_read(), Some(READ_LIMIT));
        let mut tries = 0;
        let network = tried_twice(|| {
            tries += 1;
            Err(Failed::Network("u".into(), "stalled".into()))
        });
        assert!(network.is_err() && tries == 2);
        tries = 0;
        let _ = tried_twice(|| {
            tries += 1;
            Err(Failed::Unsafe("wrong".into()))
        });
        assert_eq!(tries, 1, "a refusal is not tried again");
    }

    #[test]
    fn paths_that_leave_the_folder_are_refused() {
        assert_eq!(
            inside_path(Path::new("uv-x86_64/uv"), 1).unwrap(),
            Some(PathBuf::from("uv"))
        );
        assert_eq!(inside_path(Path::new("uv-x86_64/"), 1).unwrap(), None);
        assert_eq!(
            inside_path(Path::new("./a/b"), 0).unwrap(),
            Some(PathBuf::from("a/b"))
        );
        for bad in ["../x", "a/../../x", "/etc/passwd"] {
            assert!(inside_path(Path::new(bad), 0).is_err(), "{bad}");
        }
        assert!(safe_link_target(Path::new("libggml.so.0")));
        assert!(!safe_link_target(Path::new("../../etc/passwd")));
        assert!(!safe_link_target(Path::new("/etc/passwd")));
    }

    fn tar_gz(folder: &Path, entries: &[(&str, &[u8])], link: Option<(&str, &str)>) -> PathBuf {
        let path = folder.join("a.tar.gz");
        let file = File::create(&path).unwrap();
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::fast(),
        ));
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            builder.append_data(&mut header, name, *data).unwrap();
        }
        if let Some((name, target)) = link {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_size(0);
            builder.append_link(&mut header, name, target).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap();
        path
    }

    #[test]
    fn a_tar_gz_unpacks_with_strip_and_refuses_an_escaping_link() {
        let folder = temporary("tar");
        let archive = tar_gz(
            &folder,
            &[("uv-x/uv", b"binary"), ("uv-x/uvx", b"other")],
            Some(("uv-x/lib.so", "lib.so.1")),
        );
        let into = folder.join("out");
        unpacked(&archive, Archive::TarGz, &into, 1).unwrap();
        assert_eq!(std::fs::read(into.join("uv")).unwrap(), b"binary");
        #[cfg(unix)]
        assert_eq!(
            std::fs::read_link(into.join("lib.so")).unwrap(),
            PathBuf::from("lib.so.1")
        );
        let bad = tar_gz(
            &folder,
            &[("x/ok", b"1")],
            Some(("x/evil", "../../outside")),
        );
        let refused = unpacked(&bad, Archive::TarGz, &folder.join("out2"), 1).unwrap_err();
        assert!(matches!(refused, Failed::Unsafe(_)), "{refused}");
        std::fs::remove_dir_all(&folder).unwrap();
    }

    fn zip_with(folder: &Path, names: &[&str]) -> PathBuf {
        let path = folder.join("a.zip");
        let mut writer = zip::ZipWriter::new(File::create(&path).unwrap());
        for name in names {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(name.as_bytes()).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    #[test]
    fn a_zip_unpacks_and_refuses_a_path_out_of_the_folder() {
        let folder = temporary("zip");
        let archive = zip_with(&folder, &["uv.exe", "uvx.exe"]);
        let into = folder.join("out");
        unpacked(&archive, Archive::Zip, &into, 0).unwrap();
        assert_eq!(std::fs::read(into.join("uv.exe")).unwrap(), b"uv.exe");
        let bad = zip_with(&folder, &["ok.dll", "../evil.dll"]);
        assert!(matches!(
            unpacked(&bad, Archive::Zip, &folder.join("out2"), 0),
            Err(Failed::Unsafe(_))
        ));
        assert!(!folder.join("evil.dll").exists());
        std::fs::remove_dir_all(&folder).unwrap();
    }

    /// The real address of the smallest pinned file, by its sum (needs the network; run by hand or in CI with
    /// `cargo test -- --ignored`).
    #[test]
    #[ignore]
    fn fetches_a_pinned_file_over_tls() {
        let folder = temporary("network");
        let windows = crate::platform::target(
            crate::platform::System::Windows,
            crate::platform::Processor::X86_64,
        );
        let licence = windows.unwrap().llama_licence.unwrap();
        fetched(&licence, &folder.join("LICENSE")).unwrap();
        assert!(std::fs::read_to_string(folder.join("LICENSE"))
            .unwrap()
            .starts_with("MIT License"));
        std::fs::remove_dir_all(&folder).unwrap();
    }
}
