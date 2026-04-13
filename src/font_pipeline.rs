use std::{env, path::PathBuf, sync::Arc};

use fontdb::{Database, Source as DbSource};
use image::{GrayImage, Luma};
use swash::{
    FontRef,
    scale::{
        Render as GlyphRender, ScaleContext, Source as GlyphSource, StrikeWith, image::Content,
    },
    shape::cluster::Glyph,
    shape::{Direction, ShapeContext},
    text::{BidiClass, Codepoint, Script},
};

use crate::{
    cache::AnalysisCache,
    models::{
        AnalysisRequest, ExcludedFace, ExclusionReason, FontFaceMetadata, FontSample, RenderSpec,
        stretch_numeric_value,
    },
};

#[derive(Clone, Debug)]
pub struct FontPipeline {
    database: Database,
}

#[derive(Clone)]
struct ClusterLayout {
    glyphs: Vec<Glyph>,
    advance: f32,
}

struct RasterizedGlyph {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
    content: Content,
    data: Vec<u8>,
}

struct LayoutRaster {
    glyphs: Vec<RasterizedGlyph>,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl LayoutRaster {
    fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    fn height(&self) -> f32 {
        self.max_y - self.min_y
    }
}

impl FontPipeline {
    pub fn new() -> Self {
        let mut database = Database::new();
        for dir in font_directories() {
            if dir.exists() {
                database.load_fonts_dir(dir);
            }
        }
        Self { database }
    }

    pub fn collect_faces(&self) -> Vec<FontFaceMetadata> {
        let mut faces = self
            .database
            .faces()
            .map(|face| FontFaceMetadata {
                face_id: face.id,
                family: face
                    .families
                    .first()
                    .map(|family| family.0.clone())
                    .unwrap_or_else(|| "Unknown".to_string()),
                style: format!("{:?}", face.style),
                style_kind: face.style.into(),
                weight_value: face.weight.0,
                stretch_value: stretch_numeric_value(face.stretch),
                post_script_name: face.post_script_name.clone(),
                source_path: match &face.source {
                    DbSource::File(path) => Some(path.clone()),
                    DbSource::SharedFile(path, _) => Some(path.clone()),
                    DbSource::Binary(_) => None,
                },
                face_index: face.index,
                monospaced: face.monospaced,
            })
            .collect::<Vec<_>>();

        faces.sort_by(|left, right| {
            left.family
                .cmp(&right.family)
                .then(left.style.cmp(&right.style))
                .then(left.post_script_name.cmp(&right.post_script_name))
                .then(left.face_index.cmp(&right.face_index))
        });
        faces
    }

    pub fn render_samples(
        &self,
        faces: &[FontFaceMetadata],
        request: &AnalysisRequest,
        cache: &AnalysisCache,
    ) -> (Vec<FontSample>, Vec<ExcludedFace>) {
        let mut samples = Vec::new();
        let mut excluded = Vec::new();

        for face in faces {
            match self.render_face(face, request, cache) {
                Ok(sample) => samples.push(sample),
                Err(reason) => excluded.push(ExcludedFace {
                    face: face.clone(),
                    reason,
                }),
            }
        }

        (samples, excluded)
    }

    fn render_face(
        &self,
        face: &FontFaceMetadata,
        request: &AnalysisRequest,
        cache: &AnalysisCache,
    ) -> std::result::Result<FontSample, ExclusionReason> {
        if let Some(image) = cache.rendered(&request.text, face.face_id, &request.render_spec) {
            return Ok(FontSample {
                face: face.clone(),
                image,
            });
        }

        let rendered = self
            .database
            .with_face_data(face.face_id, |font_data, face_index| {
                render_text_to_image(
                    font_data,
                    face_index as usize,
                    &request.text,
                    &request.render_spec,
                )
            })
            .ok_or(ExclusionReason::LoadFailed)??;

        let rendered = Arc::new(rendered);
        cache.put_rendered(
            &request.text,
            face.face_id,
            &request.render_spec,
            rendered.clone(),
        );

        Ok(FontSample {
            face: face.clone(),
            image: rendered,
        })
    }
}

impl Default for FontPipeline {
    fn default() -> Self {
        Self::new()
    }
}

fn font_directories() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("/System/Library/Fonts"),
        PathBuf::from("/Library/Fonts"),
    ];
    if let Some(home) = env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join("Library/Fonts"));
    }
    dirs
}

fn render_text_to_image(
    font_data: &[u8],
    face_index: usize,
    text: &str,
    spec: &RenderSpec,
) -> std::result::Result<GrayImage, ExclusionReason> {
    if text.trim().is_empty() {
        return Err(ExclusionReason::EmptyText);
    }

    let font = FontRef::from_index(font_data, face_index).ok_or(ExclusionReason::LoadFailed)?;
    ensure_text_supported(&font, text)?;

    let mut best: Option<LayoutRaster> = None;
    let available_width = spec.width.saturating_sub(spec.padding * 2) as f32;
    let available_height = spec.height.saturating_sub(spec.padding * 2) as f32;

    let mut low = spec.min_font_size.max(1);
    let mut high = spec.max_font_size.max(low);

    while low <= high {
        let candidate_size = low + (high - low) / 2;
        match rasterize_layout(font, text, candidate_size as f32) {
            Ok(layout)
                if layout.width() > 0.0
                    && layout.height() > 0.0
                    && layout.width() <= available_width
                    && layout.height() <= available_height =>
            {
                best = Some(layout);
                low = candidate_size + 1;
            }
            Ok(_) | Err(_) => {
                if candidate_size == 0 {
                    break;
                }
                high = candidate_size.saturating_sub(1);
            }
        }
    }

    let layout = best.ok_or(ExclusionReason::FailedToFit)?;
    compose_centered_image(layout, spec)
}

fn ensure_text_supported(
    font: &FontRef<'_>,
    text: &str,
) -> std::result::Result<(), ExclusionReason> {
    let charmap = font.charmap();
    let complete = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .all(|ch| charmap.map(ch) != 0);
    if complete {
        Ok(())
    } else {
        Err(ExclusionReason::MissingGlyphs)
    }
}

fn rasterize_layout(
    font: FontRef<'_>,
    text: &str,
    size: f32,
) -> std::result::Result<LayoutRaster, ExclusionReason> {
    let script = detect_script(text);
    let direction = detect_direction(text);

    let mut shape_context = ShapeContext::new();
    let mut shaper = shape_context
        .builder(font)
        .script(script)
        .direction(direction)
        .size(size)
        .build();

    let baseline = shaper.metrics().ascent.max(0.0);
    shaper.add_str(text);

    let mut clusters = Vec::new();
    shaper.shape_with(|cluster| {
        if !cluster.is_empty() {
            clusters.push(ClusterLayout {
                glyphs: cluster.glyphs.to_vec(),
                advance: cluster.advance(),
            });
        }
    });

    if clusters.is_empty() {
        return Err(ExclusionReason::EmptyImage);
    }

    let total_advance = clusters.iter().map(|cluster| cluster.advance).sum::<f32>();
    let sources = [
        GlyphSource::ColorOutline(0),
        GlyphSource::ColorBitmap(StrikeWith::BestFit),
        GlyphSource::Outline,
    ];
    let renderer = GlyphRender::new(&sources);

    let mut scale_context = ScaleContext::new();
    let mut scaler = scale_context.builder(font).size(size).hint(true).build();

    let mut rasterized = Vec::new();
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    let mut pen_x = if direction == Direction::RightToLeft {
        total_advance
    } else {
        0.0
    };

    if direction == Direction::RightToLeft {
        for cluster in clusters.iter().rev() {
            pen_x -= cluster.advance;
            rasterize_cluster(
                cluster,
                pen_x,
                baseline,
                &renderer,
                &mut scaler,
                &mut rasterized,
                &mut min_x,
                &mut min_y,
                &mut max_x,
                &mut max_y,
            );
        }
    } else {
        for cluster in &clusters {
            rasterize_cluster(
                cluster,
                pen_x,
                baseline,
                &renderer,
                &mut scaler,
                &mut rasterized,
                &mut min_x,
                &mut min_y,
                &mut max_x,
                &mut max_y,
            );
            pen_x += cluster.advance;
        }
    }

    if rasterized.is_empty() || !min_x.is_finite() || !min_y.is_finite() {
        return Err(ExclusionReason::EmptyImage);
    }

    Ok(LayoutRaster {
        glyphs: rasterized,
        min_x,
        min_y,
        max_x,
        max_y,
    })
}

#[allow(clippy::too_many_arguments)]
fn rasterize_cluster(
    cluster: &ClusterLayout,
    cluster_x: f32,
    baseline: f32,
    renderer: &GlyphRender<'_>,
    scaler: &mut swash::scale::Scaler<'_>,
    rasterized: &mut Vec<RasterizedGlyph>,
    min_x: &mut f32,
    min_y: &mut f32,
    max_x: &mut f32,
    max_y: &mut f32,
) {
    for glyph in &cluster.glyphs {
        let Some(image) = renderer.render(scaler, glyph.id) else {
            continue;
        };

        if image.placement.width == 0 || image.placement.height == 0 {
            continue;
        }

        let left = cluster_x + glyph.x + image.placement.left as f32;
        let top = baseline - glyph.y - image.placement.top as f32;
        let right = left + image.placement.width as f32;
        let bottom = top + image.placement.height as f32;

        *min_x = min_x.min(left);
        *min_y = min_y.min(top);
        *max_x = max_x.max(right);
        *max_y = max_y.max(bottom);

        rasterized.push(RasterizedGlyph {
            left: left.round() as i32,
            top: top.round() as i32,
            width: image.placement.width,
            height: image.placement.height,
            content: image.content,
            data: image.data,
        });
    }
}

fn compose_centered_image(
    layout: LayoutRaster,
    spec: &RenderSpec,
) -> std::result::Result<GrayImage, ExclusionReason> {
    let mut image = GrayImage::from_pixel(spec.width, spec.height, Luma([spec.background]));
    let offset_x = ((spec.width as f32 - layout.width()) / 2.0 - layout.min_x).round() as i32;
    let offset_y = ((spec.height as f32 - layout.height()) / 2.0 - layout.min_y).round() as i32;

    for glyph in layout.glyphs {
        let target_left = glyph.left + offset_x;
        let target_top = glyph.top + offset_y;
        let width = glyph.width as usize;

        for row in 0..glyph.height {
            for col in 0..glyph.width {
                let x = target_left + col as i32;
                let y = target_top + row as i32;
                if x < 0 || y < 0 || x >= spec.width as i32 || y >= spec.height as i32 {
                    continue;
                }

                let source_index = row as usize * width + col as usize;
                let source = glyph_pixel(&glyph, source_index);
                if source == 0 {
                    continue;
                }

                let scaled = ((source as u16 * spec.foreground as u16) / 255) as u8;
                let target = image.get_pixel_mut(x as u32, y as u32);
                target.0[0] = target.0[0].max(scaled);
            }
        }
    }

    let has_ink = image.pixels().any(|pixel| pixel.0[0] != spec.background);
    if has_ink {
        Ok(image)
    } else {
        Err(ExclusionReason::EmptyImage)
    }
}

fn glyph_pixel(glyph: &RasterizedGlyph, index: usize) -> u8 {
    match glyph.content {
        Content::Mask => glyph.data[index],
        Content::SubpixelMask | Content::Color => {
            let base = index * 4;
            let rgb = glyph.data[base]
                .max(glyph.data[base + 1])
                .max(glyph.data[base + 2]);
            rgb.max(glyph.data[base + 3])
        }
    }
}

fn detect_script(text: &str) -> Script {
    text.chars()
        .map(Codepoint::script)
        .find(|script| !matches!(script, Script::Common | Script::Inherited | Script::Unknown))
        .unwrap_or(Script::Latin)
}

fn detect_direction(text: &str) -> Direction {
    for ch in text.chars() {
        match ch.bidi_class() {
            BidiClass::AL | BidiClass::R => return Direction::RightToLeft,
            BidiClass::L => return Direction::LeftToRight,
            _ => {}
        }
    }
    Direction::LeftToRight
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AnalysisRequest, MODEL_REVISION, RenderSpec};

    #[cfg(target_os = "macos")]
    #[test]
    fn system_scan_finds_faces() {
        let pipeline = FontPipeline::new();
        let faces = pipeline.collect_faces();
        assert!(!faces.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn system_scan_finds_collection_faces() {
        let pipeline = FontPipeline::new();
        let faces = pipeline.collect_faces();
        assert!(faces.iter().any(|face| face.face_index > 0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn renders_fixed_size_preview() -> anyhow::Result<()> {
        let pipeline = FontPipeline::new();
        let request = AnalysisRequest::new("Hello", RenderSpec::default(), MODEL_REVISION);
        let cache = AnalysisCache::default();

        let sample = pipeline
            .collect_faces()
            .into_iter()
            .find_map(|face| pipeline.render_face(&face, &request, &cache).ok())
            .expect("at least one renderable face");

        assert_eq!(sample.image.width(), 224);
        assert_eq!(sample.image.height(), 224);
        assert!(sample.image.pixels().any(|pixel| pixel.0[0] > 0));
        Ok(())
    }
}
