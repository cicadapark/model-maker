//! The card's readings for the status fold: one long-running `nvidia-smi` (by the full path `check` uses) printing a
//! line every 5 s. A value it does not give is "not known"; no card or no tool is said as such (plan section 4).

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use model_maker_launcher::{check, platform};

pub const QUERY: &str = "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total";
pub const EVERY_SECONDS: &str = "5";

#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub name: String,
    pub load_percent: Option<u32>,
    pub temperature_c: Option<u32>,
    pub memory_used_mib: Option<u64>,
    pub memory_total_mib: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Card {
    /// Not read yet: the first line comes within seconds.
    Waiting,
    /// No nvidia-smi on this PC (no NVIDIA card, or a Mac).
    NoReading,
    Read(Reading),
}

/// A number from nvidia-smi's CSV ("37", "37.5" from some Windows drivers); "[N/A]" and the like are not known.
fn number(field: &str) -> Option<f64> {
    field
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|n| n.is_finite() && *n >= 0.0)
}

/// One line of the query above, `nounits`.
pub fn parse(line: &str) -> Option<Reading> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    let [name, load, temperature, used, total] = fields[..] else {
        return None;
    };
    if name.is_empty() {
        return None;
    }
    Some(Reading {
        name: name.to_string(),
        load_percent: number(load).map(|n| n.round() as u32),
        temperature_c: number(temperature).map(|n| n.round() as u32),
        memory_used_mib: number(used).map(|n| n.round() as u64),
        memory_total_mib: number(total).map(|n| n.round() as u64),
    })
}

fn known<T: std::fmt::Display>(value: Option<T>, unit: &str) -> String {
    value.map_or_else(|| "not known".to_string(), |v| format!("{v}{unit}"))
}

/// The fold's rows: what, value.
pub fn rows(card: &Card) -> Vec<(&'static str, String)> {
    match card {
        Card::Waiting => vec![("Card", "reading…".to_string())],
        Card::NoReading => vec![("Card", "no NVIDIA card reading on this PC".to_string())],
        Card::Read(r) => vec![
            ("Card", r.name.clone()),
            ("Load", known(r.load_percent, " %")),
            ("Temperature", known(r.temperature_c, " °C")),
            (
                "Memory",
                match (r.memory_used_mib, r.memory_total_mib) {
                    (Some(used), Some(total)) => format!("{used} MB of {total} MB"),
                    (used, _) => format!("{} of not known", known(used, " MB")),
                },
            ),
        ],
    }
}

/// Watches the card in the background; ends its nvidia-smi when dropped.
pub struct Watch {
    pub latest: Arc<Mutex<Card>>,
    process: Option<std::process::Child>,
}

impl Watch {
    pub fn start() -> Watch {
        let latest = Arc::new(Mutex::new(Card::Waiting));
        let (system, _) = platform::here();
        let mut command = Command::new(check::nvidia_smi(system));
        command
            .args([QUERY, "--format=csv,noheader,nounits", "-l", EVERY_SECONDS])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        platform::quiet(&mut command);
        let mut process = match command.spawn() {
            Ok(process) => process,
            Err(_) => {
                *latest.lock().unwrap_or_else(|e| e.into_inner()) = Card::NoReading;
                return Watch {
                    latest,
                    process: None,
                };
            }
        };
        if let Some(out) = process.stdout.take() {
            let shared = latest.clone();
            std::thread::spawn(move || {
                // Two cards print two lines a round: the first card named is the one shown (plan section 10.3).
                let mut first_name: Option<String> = None;
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    let Some(reading) = parse(&line) else {
                        continue;
                    };
                    if first_name.get_or_insert_with(|| reading.name.clone()) != &reading.name {
                        continue;
                    }
                    *shared.lock().unwrap_or_else(|e| e.into_inner()) = Card::Read(reading);
                }
                let mut card = shared.lock().unwrap_or_else(|e| e.into_inner());
                if *card == Card::Waiting {
                    *card = Card::NoReading;
                }
            });
        }
        Watch {
            latest,
            process: Some(process),
        }
    }

    pub fn now(&self) -> Card {
        self.latest
            .lock()
            .map(|c| c.clone())
            .unwrap_or(Card::NoReading)
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        if let Some(process) = &mut self.process {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_with_every_value() {
        assert_eq!(
            parse("NVIDIA GeForce RTX 3060, 97, 71, 11012, 12288"),
            Some(Reading {
                name: "NVIDIA GeForce RTX 3060".into(),
                load_percent: Some(97),
                temperature_c: Some(71),
                memory_used_mib: Some(11012),
                memory_total_mib: Some(12288),
            })
        );
    }

    #[test]
    fn not_available_values_are_not_known_and_decimals_are_read() {
        let r = parse("NVIDIA GeForce RTX 3080 Laptop GPU, [N/A], 54.0, 812.5, [N/A]").unwrap();
        assert_eq!(r.load_percent, None);
        assert_eq!(r.temperature_c, Some(54));
        assert_eq!(r.memory_used_mib, Some(813));
        let shown = rows(&Card::Read(r));
        assert_eq!(shown[1].1, "not known");
        assert_eq!(shown[3].1, "813 MB of not known");
    }

    #[test]
    fn broken_lines_and_no_card_are_said() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("No devices were found"), None);
        assert_eq!(parse(", 1, 2, 3, 4"), None);
        assert_eq!(
            rows(&Card::NoReading)[0].1,
            "no NVIDIA card reading on this PC"
        );
    }
}
