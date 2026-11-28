//! egui's triangles drawn on the processor (plan section 2: the card stays free for the make). Textures are kept
//! as egui hands them over; each triangle is filled pixel by pixel, its colour times its texture, blended over what
//! is there (premultiplied alpha, as egui's own back ends).

use std::collections::HashMap;

use egui::epaint::{ClippedPrimitive, ImageData, ImageDelta, Primitive, Vertex};
use egui::{Color32, TextureId, TexturesDelta};

struct Texture {
    width: usize,
    height: usize,
    pixels: Vec<Color32>,
}

/// A picture of the window, row by row, as egui's premultiplied colours.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Color32>,
}

impl Canvas {
    /// For the tests' pictures; the window itself uses `try_new`.
    #[cfg(test)]
    pub fn new(width: usize, height: usize, background: Color32) -> Canvas {
        Canvas {
            width,
            height,
            pixels: vec![background; width * height],
        }
    }

    /// As `new`, but None when the computer cannot give the memory: the window then skips one picture instead of
    /// ending (Rust ends a program whose allocation fails). The Windows card test's seventh try (28 Nov 2026): the
    /// make's Python took all of the test PC's memory and the window vanished with it.
    pub fn try_new(width: usize, height: usize, background: Color32) -> Option<Canvas> {
        let count = width.checked_mul(height)?;
        let mut pixels = Vec::new();
        pixels.try_reserve_exact(count).ok()?;
        pixels.resize(count, background);
        Some(Canvas {
            width,
            height,
            pixels,
        })
    }

    /// The pixels as 0RGB words, softbuffer's format.
    pub fn write_0rgb(&self, out: &mut [u32]) {
        for (word, pixel) in out.iter_mut().zip(&self.pixels) {
            *word =
                (u32::from(pixel.r()) << 16) | (u32::from(pixel.g()) << 8) | u32::from(pixel.b());
        }
    }
}

#[derive(Default)]
pub struct Painter {
    textures: HashMap<TextureId, Texture>,
}

/// Which side of the line a→b the point p lies (twice the signed area).
fn edge(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// A pixel centre exactly on an edge shared by two triangles belongs to one of them only (else a see-through shape
/// would show a line along its diagonal): this is true for exactly one of a→b and b→a.
fn owns_edge(a: (f32, f32), b: (f32, f32)) -> bool {
    b.1 > a.1 || (b.1 == a.1 && b.0 < a.0)
}

fn inside(weight: f32, a: (f32, f32), b: (f32, f32)) -> bool {
    weight > 0.0 || (weight == 0.0 && owns_edge(a, b))
}

fn mul(a: u8, b: u8) -> u8 {
    ((u16::from(a) * u16::from(b) + 127) / 255) as u8
}

fn over(source: Color32, below: Color32) -> Color32 {
    let keep = 255 - source.a();
    let channel = |s: u8, d: u8| s.saturating_add(mul(d, keep));
    Color32::from_rgba_premultiplied(
        channel(source.r(), below.r()),
        channel(source.g(), below.g()),
        channel(source.b(), below.b()),
        channel(source.a(), below.a()),
    )
}

impl Painter {
    pub fn update_textures(&mut self, delta: &TexturesDelta) {
        for (id, images) in &delta.set {
            for image in images {
                self.set_texture(*id, image);
            }
        }
    }

    pub fn free_textures(&mut self, delta: &TexturesDelta) {
        for id in &delta.free {
            self.textures.remove(id);
        }
    }

    fn set_texture(&mut self, id: TextureId, delta: &ImageDelta) {
        let ImageData::Color(image) = &delta.image;
        let [width, height] = image.size;
        match delta.pos {
            None => {
                self.textures.insert(
                    id,
                    Texture {
                        width,
                        height,
                        pixels: image.pixels.clone(),
                    },
                );
            }
            Some([left, top]) => {
                let Some(texture) = self.textures.get_mut(&id) else {
                    return;
                };
                for row in 0..height {
                    for column in 0..width {
                        let (x, y) = (left + column, top + row);
                        if x < texture.width && y < texture.height {
                            texture.pixels[y * texture.width + x] =
                                image.pixels[row * width + column];
                        }
                    }
                }
            }
        }
    }

    pub fn paint(
        &self,
        canvas: &mut Canvas,
        primitives: &[ClippedPrimitive],
        pixels_per_point: f32,
    ) {
        for primitive in primitives {
            let Primitive::Mesh(mesh) = &primitive.primitive else {
                continue;
            };
            let clip = primitive.clip_rect;
            let bounds = (
                (clip.min.x * pixels_per_point).round().max(0.0) as usize,
                (clip.min.y * pixels_per_point).round().max(0.0) as usize,
                ((clip.max.x * pixels_per_point).round().max(0.0) as usize).min(canvas.width),
                ((clip.max.y * pixels_per_point).round().max(0.0) as usize).min(canvas.height),
            );
            let texture = self.textures.get(&mesh.texture_id);
            for corners in mesh.indices.as_chunks::<3>().0 {
                let vertex = |i: usize| mesh.vertices.get(corners[i] as usize);
                if let (Some(a), Some(b), Some(c)) = (vertex(0), vertex(1), vertex(2)) {
                    triangle(canvas, bounds, [a, b, c], texture, pixels_per_point);
                }
            }
        }
    }
}

fn sample(texture: Option<&Texture>, u: f32, v: f32) -> Color32 {
    let Some(texture) = texture else {
        return Color32::WHITE;
    };
    if texture.width == 0 || texture.height == 0 {
        return Color32::WHITE;
    }
    let x = ((u * texture.width as f32) as isize).clamp(0, texture.width as isize - 1) as usize;
    let y = ((v * texture.height as f32) as isize).clamp(0, texture.height as isize - 1) as usize;
    texture.pixels[y * texture.width + x]
}

fn triangle(
    canvas: &mut Canvas,
    (left, top, right, bottom): (usize, usize, usize, usize),
    mut corners: [&Vertex; 3],
    texture: Option<&Texture>,
    pixels_per_point: f32,
) {
    let at = |v: &Vertex| (v.pos.x * pixels_per_point, v.pos.y * pixels_per_point);
    let mut p = corners.map(at);
    let mut area = edge(p[0], p[1], p[2]);
    if area == 0.0 || !area.is_finite() {
        return;
    }
    if area < 0.0 {
        corners.swap(1, 2);
        p.swap(1, 2);
        area = -area;
    }
    let lowest =
        |f: fn(f32, f32) -> f32, pick: fn(&(f32, f32)) -> f32| p.iter().map(pick).fold(f32::NAN, f);
    let x0 = (lowest(f32::min, |q| q.0).floor().max(0.0) as usize).max(left);
    let y0 = (lowest(f32::min, |q| q.1).floor().max(0.0) as usize).max(top);
    let x1 = (lowest(f32::max, |q| q.0).ceil().max(0.0) as usize).min(right);
    let y1 = (lowest(f32::max, |q| q.1).ceil().max(0.0) as usize).min(bottom);
    for y in y0..y1 {
        for x in x0..x1 {
            let centre = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(p[1], p[2], centre);
            let w1 = edge(p[2], p[0], centre);
            let w2 = edge(p[0], p[1], centre);
            if !(inside(w0, p[1], p[2]) && inside(w1, p[2], p[0]) && inside(w2, p[0], p[1])) {
                continue;
            }
            let (w0, w1, w2) = (w0 / area, w1 / area, w2 / area);
            let mix = |f: fn(&Vertex) -> f32| {
                w0 * f(corners[0]) + w1 * f(corners[1]) + w2 * f(corners[2])
            };
            let channel = |f: fn(&Vertex) -> f32| mix(f).round().clamp(0.0, 255.0) as u8;
            let colour = Color32::from_rgba_premultiplied(
                channel(|v| f32::from(v.color.r())),
                channel(|v| f32::from(v.color.g())),
                channel(|v| f32::from(v.color.b())),
                channel(|v| f32::from(v.color.a())),
            );
            let texel = sample(texture, mix(|v| v.uv.x), mix(|v| v.uv.y));
            let source = Color32::from_rgba_premultiplied(
                mul(colour.r(), texel.r()),
                mul(colour.g(), texel.g()),
                mul(colour.b(), texel.b()),
                mul(colour.a(), texel.a()),
            );
            let index = y * canvas.width + x;
            canvas.pixels[index] = over(source, canvas.pixels[index]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::epaint::{Mesh, Rect};
    use egui::pos2;

    #[test]
    fn a_picture_the_computer_has_no_memory_for_is_none_not_the_end() {
        let small = Canvas::try_new(4, 3, Color32::RED).unwrap();
        assert_eq!((small.width, small.height, small.pixels.len()), (4, 3, 12));
        assert!(small.pixels.iter().all(|p| *p == Color32::RED));
        // 2^60 pixels (4 EiB): the allocation fails, and that is a None here, not an abort.
        assert!(Canvas::try_new(1 << 40, 1 << 20, Color32::RED).is_none());
        assert!(Canvas::try_new(usize::MAX, 2, Color32::RED).is_none());
    }

    fn square(colour: Color32, from: f32, to: f32) -> ClippedPrimitive {
        let mut mesh = Mesh::default();
        mesh.add_colored_rect(Rect::from_min_max(pos2(from, from), pos2(to, to)), colour);
        ClippedPrimitive {
            clip_rect: Rect::EVERYTHING,
            primitive: Primitive::Mesh(mesh),
        }
    }

    #[test]
    fn a_square_fills_exactly_its_pixels() {
        let mut canvas = Canvas::new(8, 8, Color32::BLACK);
        Painter::default().paint(&mut canvas, &[square(Color32::RED, 2.0, 6.0)], 1.0);
        let red = canvas.pixels.iter().filter(|p| **p == Color32::RED).count();
        assert_eq!(red, 16);
        assert_eq!(canvas.pixels[2 * 8 + 2], Color32::RED);
        assert_eq!(canvas.pixels[8 + 1], Color32::BLACK);
    }

    #[test]
    fn a_see_through_square_has_no_line_on_its_diagonal() {
        let mut canvas = Canvas::new(8, 8, Color32::BLACK);
        let half = Color32::from_rgba_premultiplied(100, 0, 0, 128);
        Painter::default().paint(&mut canvas, &[square(half, 0.0, 8.0)], 1.0);
        assert!(
            canvas.pixels.iter().all(|p| *p == canvas.pixels[0]),
            "every pixel blended once"
        );
        assert_eq!(canvas.pixels[0].r(), 100);
    }

    #[test]
    fn the_clip_rectangle_is_kept_and_the_scale_applied() {
        let mut canvas = Canvas::new(10, 10, Color32::BLACK);
        let mut clipped = square(Color32::GREEN, 0.0, 5.0);
        clipped.clip_rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(2.0, 5.0));
        Painter::default().paint(&mut canvas, &[clipped], 2.0);
        let green = canvas
            .pixels
            .iter()
            .filter(|p| **p == Color32::GREEN)
            .count();
        assert_eq!(
            green,
            4 * 10,
            "2 points wide at 2 pixels a point, 10 pixels high"
        );
    }

    #[test]
    fn a_texture_is_multiplied_and_patched() {
        let mut painter = Painter::default();
        let id = TextureId::Managed(0);
        let image = |w: usize, h: usize, c: Color32| {
            ImageData::Color(std::sync::Arc::new(egui::ColorImage::new(
                [w, h],
                vec![c; w * h],
            )))
        };
        let mut delta = TexturesDelta::default();
        delta.set.insert(
            id,
            [ImageDelta::full(
                image(2, 2, Color32::from_gray(128)),
                Default::default(),
            )]
            .into_iter()
            .collect(),
        );
        painter.update_textures(&delta);
        delta.clear();
        let mut patch = TexturesDelta::default();
        patch.set.insert(
            id,
            [ImageDelta::partial(
                [1, 1],
                image(1, 1, Color32::WHITE),
                Default::default(),
            )]
            .into_iter()
            .collect(),
        );
        painter.update_textures(&patch);
        patch.clear();
        assert_eq!(
            sample(painter.textures.get(&id), 0.75, 0.75),
            Color32::WHITE
        );
        assert_eq!(
            sample(painter.textures.get(&id), 0.25, 0.25),
            Color32::from_gray(128)
        );
        let mut out = [0u32; 1];
        Canvas::new(1, 1, Color32::from_rgb(1, 2, 3)).write_0rgb(&mut out);
        assert_eq!(out[0], 0x010203);
    }
}
