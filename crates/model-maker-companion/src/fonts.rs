//! The system's own font, read at start (plan section 2): nothing is shipped, so no font licence is in our archive.

use std::sync::Arc;

use egui::{FontData, FontDefinitions, FontFamily};

/// Plain sans-serif faces, in the order tried: each system's usual one first.
pub const FAMILIES: &[&str] = &[
    "Segoe UI",
    "Helvetica Neue",
    "Helvetica",
    "Noto Sans",
    "DejaVu Sans",
    "Liberation Sans",
    "Cantarell",
    "Arial",
];

/// The font bytes and face index of the first family found, else of any face the system has.
pub fn system_font() -> Option<(Vec<u8>, u32)> {
    let mut database = fontdb::Database::new();
    database.load_system_fonts();
    let normal = |database: &fontdb::Database, family: fontdb::Family| {
        database.query(&fontdb::Query {
            families: &[family],
            weight: fontdb::Weight::NORMAL,
            stretch: fontdb::Stretch::Normal,
            style: fontdb::Style::Normal,
        })
    };
    let chosen = FAMILIES
        .iter()
        .find_map(|name| normal(&database, fontdb::Family::Name(name)))
        .or_else(|| normal(&database, fontdb::Family::SansSerif))
        .or_else(|| database.faces().next().map(|face| face.id))?;
    database.with_face_data(chosen, |data, index| (data.to_vec(), index))
}

/// egui's fonts: the system's face for every family (egui's own bundled fonts are not built in).
pub fn definitions(font: Vec<u8>, index: u32) -> FontDefinitions {
    let mut definitions = FontDefinitions::empty();
    let mut data = FontData::from_owned(font);
    data.index = index;
    definitions
        .font_data
        .insert("system".to_string(), Arc::new(data));
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        definitions
            .families
            .insert(family, vec!["system".to_string()]);
    }
    definitions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_definitions_name_only_the_systems_face() {
        let made = definitions(vec![1, 2, 3], 2);
        assert_eq!(made.font_data.len(), 1);
        assert_eq!(made.font_data["system"].index, 2);
        assert_eq!(made.families[&FontFamily::Proportional], ["system"]);
    }

    /// egui's bundled fonts (their own font licences) must never come back into the build (plan section 2).
    #[test]
    fn eguis_bundled_fonts_are_not_built_in() {
        assert!(FontDefinitions::empty().font_data.is_empty());
        let lock =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))
                .unwrap();
        assert!(
            !lock.contains("name = \"epaint_default_fonts\""),
            "the fonts crate is in Cargo.lock"
        );
        let manifest =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")).unwrap();
        for line in manifest.lines().filter(|l| l.starts_with("egui")) {
            assert!(
                line.contains("default-features = false") && !line.contains("default_fonts"),
                "{line}"
            );
        }
    }
}
