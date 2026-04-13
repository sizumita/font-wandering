use std::{fmt, path::PathBuf, sync::Arc};

use fontdb::{ID, Stretch, Style};
use image::GrayImage;

pub const DEFAULT_INPUT_TEXT: &str = "Aaあ@";
pub const MODEL_REVISION: &str = "glyph-hybrid-bootstrap-v1";
pub const ENCODER_MODEL_FILENAME: &str = "glyph_encoder_bootstrap_f37072fd.pth";
pub const FEATURE_STATS_FILENAME: &str = "glyph_feature_stats_bootstrap.json";
pub const DEFAULT_IMAGE_SIZE: u32 = 224;
pub const DEFAULT_PADDING: u32 = 16;
pub const DEFAULT_MIN_CLUSTER_SIZE: usize = 5;
pub const DEFAULT_UMAP_NEIGHBORS: usize = 15;
pub const DEFAULT_PROJECTION_BINS: usize = 8;

pub type SharedGrayImage = Arc<GrayImage>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StyleKind {
    Normal,
    Italic,
    Oblique,
}

impl StyleKind {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Italic => "Italic",
            Self::Oblique => "Oblique",
        }
    }
}

impl fmt::Display for StyleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

impl From<Style> for StyleKind {
    fn from(value: Style) -> Self {
        match value {
            Style::Normal => Self::Normal,
            Style::Italic => Self::Italic,
            Style::Oblique => Self::Oblique,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct RenderSpec {
    pub width: u32,
    pub height: u32,
    pub padding: u32,
    pub foreground: u8,
    pub background: u8,
    pub min_font_size: u32,
    pub max_font_size: u32,
}

impl Default for RenderSpec {
    fn default() -> Self {
        Self {
            width: DEFAULT_IMAGE_SIZE,
            height: DEFAULT_IMAGE_SIZE,
            padding: DEFAULT_PADDING,
            foreground: 255,
            background: 0,
            min_font_size: 10,
            max_font_size: 196,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AnalysisRequest {
    pub text: String,
    pub render_spec: RenderSpec,
    pub model_revision: String,
}

impl AnalysisRequest {
    pub fn new(
        text: impl Into<String>,
        render_spec: RenderSpec,
        model_revision: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            render_spec,
            model_revision: model_revision.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FontFaceMetadata {
    pub face_id: ID,
    pub family: String,
    pub style: String,
    pub style_kind: StyleKind,
    pub weight_value: u16,
    pub stretch_value: u16,
    pub post_script_name: String,
    pub source_path: Option<PathBuf>,
    pub face_index: u32,
    pub monospaced: bool,
}

impl FontFaceMetadata {
    pub fn display_name(&self) -> String {
        format!("{} {}", self.family, self.style)
    }

    pub fn source_label(&self) -> String {
        self.source_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(in-memory)".to_string())
    }

    pub fn weight_label(&self) -> &'static str {
        match self.weight_value {
            0..=149 => "Thin",
            150..=249 => "ExtraLight",
            250..=349 => "Light",
            350..=449 => "Regular",
            450..=549 => "Medium",
            550..=649 => "Semibold",
            650..=749 => "Bold",
            750..=849 => "ExtraBold",
            _ => "Black",
        }
    }

    pub fn stretch_label(&self) -> &'static str {
        match self.stretch_value {
            1 => "UltraCondensed",
            2 => "ExtraCondensed",
            3 => "Condensed",
            4 => "SemiCondensed",
            5 => "Normal",
            6 => "SemiExpanded",
            7 => "Expanded",
            8 => "ExtraExpanded",
            9 => "UltraExpanded",
            _ => "Custom",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExclusionReason {
    EmptyText,
    LoadFailed,
    MissingGlyphs,
    FailedToFit,
    EmptyImage,
    RenderFailed(String),
}

impl fmt::Display for ExclusionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText => write!(f, "入力文字列が空です"),
            Self::LoadFailed => write!(f, "フォントデータを読み込めませんでした"),
            Self::MissingGlyphs => write!(f, "全文字列を単独 face で描画できませんでした"),
            Self::FailedToFit => write!(f, "固定サイズ内に収まりませんでした"),
            Self::EmptyImage => write!(f, "描画結果が空でした"),
            Self::RenderFailed(message) => write!(f, "{message}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExcludedFace {
    pub face: FontFaceMetadata,
    pub reason: ExclusionReason,
}

#[derive(Clone, Debug)]
pub struct FontSample {
    pub face: FontFaceMetadata,
    pub image: SharedGrayImage,
}

#[derive(Clone, Debug)]
pub struct EmbeddingPoint {
    pub face: FontFaceMetadata,
    pub image: SharedGrayImage,
    pub embedding: Arc<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct MapPoint {
    pub face: FontFaceMetadata,
    pub x: f32,
    pub y: f32,
    pub cluster_id: i32,
    pub preview: SharedGrayImage,
    pub embedding: Arc<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct AnalysisStats {
    pub scanned_faces: usize,
    pub rendered_faces: usize,
    pub excluded_faces: usize,
    pub cluster_count: usize,
    pub noise_points: usize,
    pub embedding_dim: usize,
    pub device_label: String,
}

impl Default for AnalysisStats {
    fn default() -> Self {
        Self {
            scanned_faces: 0,
            rendered_faces: 0,
            excluded_faces: 0,
            cluster_count: 0,
            noise_points: 0,
            embedding_dim: 0,
            device_label: "uninitialized".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnalysisResult {
    pub request: AnalysisRequest,
    pub points: Vec<MapPoint>,
    pub excluded_faces: Vec<ExcludedFace>,
    pub stats: AnalysisStats,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnalysisStage {
    Idle,
    ScanningFonts,
    RenderingGlyphs,
    Embedding,
    Reducing,
    Clustering,
    Finished,
    Failed,
}

impl AnalysisStage {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Idle => "待機中",
            Self::ScanningFonts => "フォント走査中",
            Self::RenderingGlyphs => "グリフ描画中",
            Self::Embedding => "Hybrid 特徴抽出中",
            Self::Reducing => "UMAP 圧縮中",
            Self::Clustering => "HDBSCAN クラスタリング中",
            Self::Finished => "完了",
            Self::Failed => "失敗",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnalysisProgress {
    pub stage: AnalysisStage,
    pub completed: usize,
    pub total: usize,
    pub detail: String,
}

impl AnalysisProgress {
    pub fn new(
        stage: AnalysisStage,
        completed: usize,
        total: usize,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            completed,
            total,
            detail: detail.into(),
        }
    }

    pub fn idle() -> Self {
        Self::new(
            AnalysisStage::Idle,
            0,
            0,
            "Analyze を押すと解析を開始します",
        )
    }

    pub fn summary(&self) -> String {
        if self.total == 0 {
            format!("{}: {}", self.stage.label(), self.detail)
        } else {
            format!(
                "{}: {}/{} {}",
                self.stage.label(),
                self.completed,
                self.total,
                self.detail
            )
        }
    }
}

pub fn stretch_numeric_value(stretch: Stretch) -> u16 {
    stretch.to_number()
}
