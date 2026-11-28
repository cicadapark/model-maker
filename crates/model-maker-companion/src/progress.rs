//! Bars and times in words. A bar never goes back; a value not known is written "not known" (plan section 1).

use std::time::Duration;

/// A bar's shown fraction: it only moves forward.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bar {
    shown: f32,
}

impl Bar {
    pub fn to(&mut self, wanted: f32) -> f32 {
        if wanted.is_finite() {
            self.shown = self.shown.max(wanted.clamp(0.0, 1.0));
        }
        self.shown
    }

    pub fn shown(&self) -> f32 {
        self.shown
    }
}

/// A stage timed by the clock against its expected time stops short of the end: it is not done until its line says so.
pub const CLOCK_BAR_MOST: f32 = 0.95;

pub fn by_clock(elapsed: Duration, expected_seconds: f64) -> f32 {
    if expected_seconds <= 0.0 {
        return 0.0;
    }
    ((elapsed.as_secs_f64() / expected_seconds) as f32).min(CLOCK_BAR_MOST)
}

pub fn duration_words(seconds: f64) -> String {
    let minutes = (seconds / 60.0).round() as u64;
    match minutes {
        _ if seconds < 60.0 => "under 1 min".to_string(),
        0..=59 => format!("{} min", minutes.max(1)),
        _ => format!("{} h {} min", minutes / 60, minutes % 60),
    }
}

/// "about N min left", or "longer than usual" past the expected time, or "not known" without one.
pub fn time_left_words(elapsed: Duration, expected_seconds: Option<f64>) -> String {
    match expected_seconds {
        None => "left: not known".to_string(),
        Some(expected) if elapsed.as_secs_f64() >= expected => "longer than usual".to_string(),
        Some(expected) => format!(
            "about {} left",
            duration_words(expected - elapsed.as_secs_f64())
        ),
    }
}

pub fn median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|v| v.is_finite() && *v >= 0.0);
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    })
}

pub fn gigabytes(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bar_never_goes_back() {
        let mut bar = Bar::default();
        assert_eq!(bar.to(0.4), 0.4);
        assert_eq!(bar.to(0.2), 0.4);
        assert_eq!(bar.to(f32::NAN), 0.4);
        assert_eq!(bar.to(1.7), 1.0);
        assert_eq!(bar.to(0.0), 1.0);
    }

    #[test]
    fn the_clock_bar_stops_short_and_times_are_words() {
        assert_eq!(by_clock(Duration::from_secs(30), 60.0), 0.5);
        assert_eq!(by_clock(Duration::from_secs(600), 60.0), CLOCK_BAR_MOST);
        assert_eq!(by_clock(Duration::from_secs(5), 0.0), 0.0);
        assert_eq!(duration_words(12.0), "under 1 min");
        assert_eq!(duration_words(185.0), "3 min");
        assert_eq!(duration_words(3900.0), "1 h 5 min");
        assert_eq!(
            time_left_words(Duration::from_secs(60), None),
            "left: not known"
        );
        assert_eq!(
            time_left_words(Duration::from_secs(60), Some(300.0)),
            "about 4 min left"
        );
        assert_eq!(
            time_left_words(Duration::from_secs(400), Some(300.0)),
            "longer than usual"
        );
    }

    #[test]
    fn the_median_of_past_times() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(vec![f64::NAN, 5.0]), Some(5.0));
        assert_eq!(gigabytes(57_300_000_000), "57.3 GB");
    }
}
