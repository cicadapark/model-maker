//! Windows only: the C++ runtime torch's and llama.cpp's libraries need, which a fresh Windows lacks. `setup` offers
//! to fetch Microsoft's Visual C++ Redistributable from Microsoft's own address, checks that its signature is
//! Microsoft's and valid (its sum cannot be pinned: Microsoft replaces the file with each update), and starts it only
//! after the person says yes; it asks for administrator rights once. Without a yes, setup stops with the one
//! instruction. Nothing is copied from other packages.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Microsoft's own permanent address of the current x64 redistributable.
pub const INSTALLER_URL: &str = "https://aka.ms/vs/17/release/vc_redist.x64.exe";
pub const INSTRUCTION: &str = "install Microsoft's Visual C++ Redistributable (x64) from \
https://aka.ms/vs/17/release/vc_redist.x64.exe (Microsoft's own address), then run `model-maker setup` again";
/// What the installer's exit codes mean: done, done but Windows wants a restart, a newer one is already there.
const DONE: i32 = 0;
const DONE_RESTART: i32 = 3010;
const NEWER_THERE: i32 = 1638;

/// The signer's and the issuer's simple names (their certificates' common names) the installer must carry, read
/// from the real file on 26 Nov 2026: Microsoft's code-signing authority's name ends with its year, so only its
/// beginning is fixed here. "Valid" means Windows built the signer's chain to a root it trusts; these names then
/// say whose certificate it is and which authority issued it.
pub const SIGNER: &str = "Microsoft Corporation";
pub const SIGNING_AUTHORITY: &str = "Microsoft Code Signing PCA";

/// What Windows says about a file's signature: its status, and the simple names of the signer's certificate and of
/// its issuer, read from the certificate's own fields (never from its display text, where a free-text field could
/// hold anything, commas and `CN=` included).
#[derive(Debug, PartialEq, Eq)]
pub struct Signature {
    pub status: String,
    pub signer: String,
    pub issuer: String,
}

/// PowerShell's answer: exactly three lines, the status, the signer's name and the issuer's name (empty when the
/// file has no signer). Anything else is no answer.
pub fn parse_signature(text: &str) -> Option<Signature> {
    let lines: Vec<&str> = text.lines().collect();
    match lines.as_slice() {
        [status, signer, issuer] if !status.is_empty() => Some(Signature {
            status: status.to_string(),
            signer: signer.to_string(),
            issuer: issuer.to_string(),
        }),
        _ => None,
    }
}

/// Valid, signed by Microsoft itself (the exact name), under Microsoft's code-signing authority.
pub fn is_microsofts(signature: &Signature) -> bool {
    signature.status == "Valid"
        && signature.signer == SIGNER
        && signature.issuer.starts_with(SIGNING_AUTHORITY)
}

fn powershell() -> PathBuf {
    PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into()))
        .join("System32")
        .join("WindowsPowerShell")
        .join("v1.0")
        .join("powershell.exe")
}

/// A path as a PowerShell single-quoted string.
pub fn quoted(path: &Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', "''"))
}

/// Windows' own verdict on the file's signature (Authenticode, through PowerShell).
pub fn signature_of(file: &Path) -> Option<Signature> {
    // GetNameInfo(SimpleName, false) is the subject's simple name, (SimpleName, true) the issuer's.
    let script = format!(
        "$s = Get-AuthenticodeSignature -LiteralPath {}; $c = $s.SignerCertificate; $s.Status.ToString(); \
         if ($c) {{ $c.GetNameInfo('SimpleName', $false); $c.GetNameInfo('SimpleName', $true) }} else {{ ''; '' }}",
        quoted(file)
    );
    let done = Command::new(powershell())
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .ok()?;
    parse_signature(&String::from_utf8_lossy(&done.stdout))
}

/// Yes only for "y" or "yes" (any case); anything else, or no answer, is no.
pub fn said_yes(question: &str, input: &mut impl BufRead, output: &mut impl Write) -> bool {
    let _ = write!(output, "{question} [y/N] ");
    let _ = output.flush();
    let mut answer = String::new();
    if input.read_line(&mut answer).is_err() {
        return false;
    }
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Installed,
    InstalledRestartWanted,
    Declined,
    Failed(String),
}

/// The installer started with administrator rights (Windows asks the person), waited for; its exit code judged.
fn started(installer: &Path) -> Result<i32, String> {
    let script = format!(
        "$p = Start-Process -FilePath {} -ArgumentList '/install','/passive','/norestart' -Verb RunAs -Wait -PassThru; \
         $p.ExitCode",
        quoted(installer)
    );
    let done = Command::new(powershell())
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .map_err(|e| e.to_string())?;
    let said = String::from_utf8_lossy(&done.stdout);
    said.lines()
        .rev()
        .find_map(|l| l.trim().parse().ok())
        .ok_or_else(|| {
            format!(
                "the installer did not start (was the administrator prompt declined?) {}",
                String::from_utf8_lossy(&done.stderr).trim()
            )
        })
}

pub fn judged_exit(code: i32) -> Outcome {
    match code {
        DONE | NEWER_THERE => Outcome::Installed,
        DONE_RESTART => Outcome::InstalledRestartWanted,
        other => Outcome::Failed(format!("the installer ended with code {other}")),
    }
}

/// Whether to fetch the runtime: a yes given elsewhere is enough; else the person's answer, asked only when someone
/// can answer at the terminal; else no.
pub fn decided(ask: bool, already_yes: bool, answer: impl FnOnce() -> bool) -> bool {
    already_yes || (ask && answer())
}

/// The whole offer: ask, fetch into `tools`, check the signature, start. `ask` is false when nobody can answer.
/// `ask`: a person can answer at the terminal; `already_yes`: they said yes elsewhere (the companion's window), so the
/// question is not asked again.
pub fn offered(tools: &Path, ask: bool, already_yes: bool) -> Outcome {
    println!("This PC has no Visual C++ runtime (msvcp140.dll), which torch and the question judge need.");
    println!("setup can fetch Microsoft's Visual C++ Redistributable (x64) from Microsoft ({INSTALLER_URL}),");
    println!("check that it is signed by Microsoft, and start it. Windows will ask for administrator rights once.");
    let stdin = std::io::stdin();
    let fetch = decided(ask, already_yes, || {
        said_yes(
            "Fetch and start it?",
            &mut stdin.lock(),
            &mut std::io::stdout(),
        )
    });
    if !fetch {
        return Outcome::Declined;
    }
    let installer = tools.join("vc_redist.x64.exe");
    if let Err(failed) = crate::download::fetched_unpinned(INSTALLER_URL, &installer) {
        return Outcome::Failed(failed.to_string());
    }
    let verdict = match signature_of(&installer) {
        Some(signature) if is_microsofts(&signature) => started(&installer).map(judged_exit),
        Some(signature) => Err(format!(
            "its signature is {} by \"{}\" under \"{}\", not Microsoft's valid one; it was not started",
            signature.status, signature.signer, signature.issuer
        )),
        None => Err("Windows gave no verdict on its signature; it was not started".to_string()),
    };
    let _ = std::fs::remove_file(&installer);
    verdict.unwrap_or_else(Outcome::Failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_runtime_is_fetched_only_on_a_yes() {
        let never = || -> bool { panic!("asked although it must not be") };
        assert!(
            !decided(false, false, never),
            "nobody to ask, no yes: declined"
        );
        assert!(decided(false, true, never), "the window's yes, no terminal");
        assert!(
            decided(true, true, never),
            "the window's yes: not asked again"
        );
        assert!(decided(true, false, || true));
        assert!(!decided(true, false, || false));
    }

    fn signed(status: &str, signer: &str, issuer: &str) -> Signature {
        Signature {
            status: status.into(),
            signer: signer.into(),
            issuer: issuer.into(),
        }
    }

    #[test]
    fn only_microsofts_valid_signature_is_accepted() {
        // What the real installer gives (read from its certificates, 26 Nov 2026).
        let real = parse_signature(
            "Valid\r\nMicrosoft Corporation\r\nMicrosoft Code Signing PCA 2011\r\n",
        )
        .unwrap();
        assert_eq!(
            real,
            signed(
                "Valid",
                "Microsoft Corporation",
                "Microsoft Code Signing PCA 2011"
            )
        );
        assert!(is_microsofts(&real));
        assert!(!is_microsofts(&Signature {
            status: "HashMismatch".into(),
            ..real
        }));
        for (signer, issuer) in [
            // A free-text field holding the maker's parts: the old display-text check took this one.
            (
                "x, CN=Microsoft Corporation, O=Microsoft Corporation, y",
                "Microsoft Code Signing PCA 2011",
            ),
            (
                "Microsoft Corporation Fake",
                "Microsoft Code Signing PCA 2011",
            ),
            ("microsoft corporation", "Microsoft Code Signing PCA 2011"),
            (" Microsoft Corporation", "Microsoft Code Signing PCA 2011"),
            ("Somebody", "Microsoft Code Signing PCA 2011"),
            ("Microsoft Corporation", "Some Public Code Signing CA"),
            ("Microsoft Corporation", "x Microsoft Code Signing PCA 2011"),
            ("Microsoft Corporation", ""),
        ] {
            assert!(
                !is_microsofts(&signed("Valid", signer, issuer)),
                "{signer} / {issuer}"
            );
        }
    }

    #[test]
    fn only_exactly_three_lines_are_an_answer() {
        assert_eq!(
            parse_signature("NotSigned\n\n\n"),
            Some(signed("NotSigned", "", ""))
        );
        assert!(!is_microsofts(&parse_signature("NotSigned\n\n\n").unwrap()));
        for text in [
            "",
            "Valid",
            "Valid\nMicrosoft Corporation",
            "Valid\nMicrosoft Corporation\nMicrosoft Code Signing PCA 2011\nextra",
            "\nMicrosoft Corporation\nMicrosoft Code Signing PCA 2011",
            // The old one-line form.
            "Valid|CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US",
        ] {
            assert_eq!(parse_signature(text), None, "{text:?}");
        }
    }

    #[test]
    fn only_a_clear_yes_is_yes() {
        let mut out = Vec::new();
        for (answer, yes) in [
            ("y\n", true),
            ("YES\r\n", true),
            ("n\n", false),
            ("\n", false),
            ("", false),
            ("yep\n", false),
        ] {
            assert_eq!(
                said_yes("Go?", &mut answer.as_bytes(), &mut out),
                yes,
                "{answer:?}"
            );
        }
        assert!(String::from_utf8(out).unwrap().contains("Go? [y/N]"));
    }

    #[test]
    fn exit_codes() {
        assert_eq!(judged_exit(0), Outcome::Installed);
        assert_eq!(judged_exit(1638), Outcome::Installed);
        assert_eq!(judged_exit(3010), Outcome::InstalledRestartWanted);
        assert!(matches!(judged_exit(1602), Outcome::Failed(_)));
    }

    #[test]
    fn paths_are_quoted_for_powershell() {
        assert_eq!(
            quoted(Path::new(r"C:\model-maker\it's\x.exe")),
            r"'C:\model-maker\it''s\x.exe'"
        );
    }
}
