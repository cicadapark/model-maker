//! `model-maker-simplify dense.mesh kept.indices --most N`: the dense surface's kept triangles, at most N.

use std::process::ExitCode;

use model_maker_simplify::{indices_bytes, read_surface, simplified};

const USAGE: &str = "usage: model-maker-simplify dense.mesh kept.indices --most TRIANGLES";

fn run(args: &[String]) -> Result<String, String> {
    let [dense, kept, flag, most] = args else {
        return Err(USAGE.into());
    };
    if flag != "--most" {
        return Err(USAGE.into());
    }
    let most: usize = most
        .parse()
        .map_err(|_| format!("--most {most}: not a count\n{USAGE}"))?;
    if most == 0 {
        return Err("--most must be at least 1".into());
    }
    let bytes = std::fs::read(dense).map_err(|e| format!("{dense}: {e}"))?;
    let surface = read_surface(&bytes).map_err(|e| format!("{dense}: {e}"))?;
    let (indices, method) = simplified(&surface, most)?;
    std::fs::write(kept, indices_bytes(&indices)).map_err(|e| format!("{kept}: {e}"))?;
    Ok(format!(
        "{} triangles kept of {} by {}",
        indices.len() / 3,
        surface.triangles(),
        method.words()
    ))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(said) => {
            println!("{said}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
