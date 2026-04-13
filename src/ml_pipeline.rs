use std::{fs, path::PathBuf, sync::Arc};

use anyhow::{Context, Result, bail};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::resnet;
use hdbscan::{Hdbscan, HdbscanHyperParams, NnAlgorithm};
use ndarray::Array2;
use serde::Deserialize;
use umap::{Umap, UmapConfig};

use crate::{
    cache::AnalysisCache,
    models::{
        AnalysisRequest, AnalysisStats, DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_UMAP_NEIGHBORS,
        ENCODER_MODEL_FILENAME, EmbeddingPoint, FEATURE_STATS_FILENAME, FontSample, MODEL_REVISION,
        MapPoint,
    },
    morphology::{extract_morphology_features, morphology_feature_count},
};

const ENCODER_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const ENCODER_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Debug, Deserialize)]
struct HybridFeatureConfig {
    learned_scale: f32,
    morphology_scale: f32,
    projection_bins: usize,
    morphology_means: Vec<f32>,
    morphology_stds: Vec<f32>,
}

impl HybridFeatureConfig {
    fn load(path: &PathBuf) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let expected = morphology_feature_count(config.projection_bins);
        if config.morphology_means.len() != expected || config.morphology_stds.len() != expected {
            bail!(
                "feature stats size mismatch: expected {expected}, got means={} stds={}",
                config.morphology_means.len(),
                config.morphology_stds.len()
            );
        }
        Ok(config)
    }
}

#[derive(Clone, Debug)]
pub struct MlPipeline {
    model_path: PathBuf,
    feature_stats_path: PathBuf,
}

impl MlPipeline {
    pub fn new() -> Self {
        let assets_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("models");
        Self {
            model_path: assets_dir.join(ENCODER_MODEL_FILENAME),
            feature_stats_path: assets_dir.join(FEATURE_STATS_FILENAME),
        }
    }

    pub fn model_revision(&self) -> &'static str {
        MODEL_REVISION
    }

    pub fn model_path(&self) -> &PathBuf {
        &self.model_path
    }

    pub fn select_device(&self) -> (Device, String) {
        select_device()
    }

    pub fn project_embeddings(&self, points: &[EmbeddingPoint]) -> Result<Vec<(f32, f32)>> {
        reduce_embeddings(points)
    }

    pub fn cluster_points(&self, points: &[EmbeddingPoint]) -> Vec<i32> {
        cluster_feature_points(points)
    }

    pub fn analyze(
        &self,
        request: &AnalysisRequest,
        samples: &[FontSample],
        cache: &AnalysisCache,
    ) -> Result<(Vec<MapPoint>, AnalysisStats)> {
        if samples.is_empty() {
            return Ok((
                Vec::new(),
                AnalysisStats {
                    scanned_faces: 0,
                    rendered_faces: 0,
                    excluded_faces: 0,
                    cluster_count: 0,
                    noise_points: 0,
                    embedding_dim: 0,
                    device_label: "CPU fallback".to_string(),
                },
            ));
        }

        let (device, device_label) = select_device();
        let embedding_points = self.embed_samples(request, samples, cache, &device)?;
        let embedding_dim = embedding_points
            .first()
            .map(|point| point.embedding.len())
            .unwrap_or(0);

        let projected = reduce_embeddings(&embedding_points)?;
        let labels = cluster_feature_points(&embedding_points);

        let points = embedding_points
            .into_iter()
            .zip(projected)
            .zip(labels)
            .map(|((point, (x, y)), cluster_id)| MapPoint {
                face: point.face,
                x,
                y,
                cluster_id,
                preview: point.image,
                embedding: point.embedding,
            })
            .collect::<Vec<_>>();

        let cluster_count = points
            .iter()
            .filter_map(|point| (point.cluster_id >= 0).then_some(point.cluster_id))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let noise_points = points.iter().filter(|point| point.cluster_id < 0).count();

        Ok((
            points,
            AnalysisStats {
                scanned_faces: samples.len(),
                rendered_faces: samples.len(),
                excluded_faces: 0,
                cluster_count,
                noise_points,
                embedding_dim,
                device_label,
            },
        ))
    }

    pub fn embed_samples(
        &self,
        request: &AnalysisRequest,
        samples: &[FontSample],
        cache: &AnalysisCache,
        device: &Device,
    ) -> Result<Vec<EmbeddingPoint>> {
        let config = HybridFeatureConfig::load(&self.feature_stats_path)?;
        let missing = samples
            .iter()
            .filter(|sample| {
                cache
                    .embedding(
                        &request.text,
                        sample.face.face_id,
                        &request.render_spec,
                        &request.model_revision,
                    )
                    .is_none()
            })
            .cloned()
            .collect::<Vec<_>>();

        if !missing.is_empty() {
            if !self.model_path.exists() {
                bail!(
                    "glyph encoder weights が見つかりません: {}",
                    self.model_path.display()
                );
            }

            let vb = VarBuilder::from_pth(&self.model_path, DType::F32, device)
                .with_context(|| format!("failed to load {}", self.model_path.display()))?;
            let model = resnet::resnet18_no_final_layer(vb)?;

            for chunk in missing.chunks(16) {
                let input_tensors = chunk
                    .iter()
                    .map(|sample| gray_image_to_tensor(sample.image.as_ref(), device))
                    .collect::<Result<Vec<_>>>()?;
                let refs = input_tensors.iter().collect::<Vec<_>>();
                let batch = Tensor::stack(&refs, 0)?;
                let learned_rows = batch.apply(&model)?.to_vec2::<f32>()?;

                for (sample, learned_row) in chunk.iter().zip(learned_rows) {
                    let morphology =
                        extract_morphology_features(sample.image.as_ref(), config.projection_bins);
                    let hybrid = compose_hybrid_feature(&learned_row, &morphology, &config)?;
                    cache.put_embedding(
                        &request.text,
                        sample.face.face_id,
                        &request.render_spec,
                        &request.model_revision,
                        Arc::new(hybrid),
                    );
                }
            }
        }

        samples
            .iter()
            .map(|sample| {
                let embedding = cache
                    .embedding(
                        &request.text,
                        sample.face.face_id,
                        &request.render_spec,
                        &request.model_revision,
                    )
                    .context("embedding cache miss after hybrid inference")?;
                Ok(EmbeddingPoint {
                    face: sample.face.clone(),
                    image: sample.image.clone(),
                    embedding,
                })
            })
            .collect::<Result<Vec<_>>>()
    }
}

impl Default for MlPipeline {
    fn default() -> Self {
        Self::new()
    }
}

fn select_device() -> (Device, String) {
    #[cfg(target_os = "macos")]
    if let Ok(device) = Device::new_metal(0) {
        return (device, "Metal".to_string());
    }

    (Device::Cpu, "CPU fallback".to_string())
}

fn gray_image_to_tensor(image: &image::GrayImage, device: &Device) -> Result<Tensor> {
    let mut channels = Vec::with_capacity((image.width() * image.height() * 3) as usize);
    for channel in 0..3 {
        for pixel in image.pixels() {
            let value = pixel.0[0] as f32 / 255.0;
            let normalized = (value - ENCODER_MEAN[channel]) / ENCODER_STD[channel];
            channels.push(normalized);
        }
    }
    Ok(Tensor::from_vec(
        channels,
        (3, image.height() as usize, image.width() as usize),
        device,
    )?)
}

fn compose_hybrid_feature(
    learned_embedding: &[f32],
    morphology: &[f32],
    config: &HybridFeatureConfig,
) -> Result<Vec<f32>> {
    let expected = morphology_feature_count(config.projection_bins);
    if morphology.len() != expected {
        bail!(
            "morphology feature count mismatch: expected {expected}, got {}",
            morphology.len()
        );
    }

    let learned_branch = normalize_branch(learned_embedding, config.learned_scale);
    let morphology_branch = morphology
        .iter()
        .zip(&config.morphology_means)
        .zip(&config.morphology_stds)
        .map(|((&value, &mean), &std)| (value - mean) / std.max(1e-6))
        .collect::<Vec<_>>();
    let morphology_branch = normalize_branch(&morphology_branch, config.morphology_scale);

    Ok(learned_branch
        .into_iter()
        .chain(morphology_branch)
        .collect())
}

fn normalize_branch(values: &[f32], branch_scale: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let l2 = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let base = if l2 > f32::EPSILON { l2 } else { 1.0 };
    let dimension_scale = branch_scale / (values.len() as f32).sqrt().max(1.0);

    values
        .iter()
        .map(|value| value / base * dimension_scale)
        .collect()
}

fn reduce_embeddings(points: &[EmbeddingPoint]) -> Result<Vec<(f32, f32)>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    if points.len() == 1 {
        return Ok(vec![(0.0, 0.0)]);
    }

    if points.len() == 2 {
        return Ok(vec![(-1.0, 0.0), (1.0, 0.0)]);
    }

    let rows = points.len();
    let cols = points
        .first()
        .map(|point| point.embedding.len())
        .unwrap_or_default();
    let flat = points
        .iter()
        .flat_map(|point| point.embedding.iter().copied())
        .collect::<Vec<_>>();
    let data = Array2::from_shape_vec((rows, cols), flat)?;

    let neighbors = DEFAULT_UMAP_NEIGHBORS.clamp(2, rows.saturating_sub(1));
    let (knn_indices, knn_dists) = build_cosine_knn(points, neighbors);
    let init = build_deterministic_init(points);

    let mut config = UmapConfig::default();
    config.n_components = 2;
    config.graph.n_neighbors = neighbors;
    config.manifold.min_dist = 0.1;

    let fitted = Umap::new(config).fit(
        data.view(),
        knn_indices.view(),
        knn_dists.view(),
        init.view(),
    );
    let embedding = fitted.embedding().to_owned();

    Ok(embedding
        .outer_iter()
        .map(|row| (row[0], row[1]))
        .collect::<Vec<_>>())
}

fn build_deterministic_init(points: &[EmbeddingPoint]) -> Array2<f32> {
    let rows = points.len();
    let cols = points
        .first()
        .map(|point| point.embedding.len())
        .unwrap_or_default();
    let mut init = Array2::zeros((rows, 2));

    if cols >= 2 {
        for (row, point) in points.iter().enumerate() {
            init[(row, 0)] = point.embedding[0];
            init[(row, 1)] = point.embedding[1];
        }
    } else if cols == 1 {
        for (row, point) in points.iter().enumerate() {
            init[(row, 0)] = point.embedding[0];
            init[(row, 1)] = 0.0;
        }
    } else {
        for (row, _) in points.iter().enumerate() {
            let angle = row as f32 / rows as f32 * std::f32::consts::TAU;
            init[(row, 0)] = angle.cos();
            init[(row, 1)] = angle.sin();
        }
        return init;
    }

    normalize_column(&mut init, 0);
    normalize_column(&mut init, 1);
    init
}

fn normalize_column(data: &mut Array2<f32>, column: usize) {
    let values = (0..data.nrows())
        .map(|row| data[(row, column)])
        .collect::<Vec<_>>();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).max(1e-6);

    for row in 0..data.nrows() {
        data[(row, column)] = ((data[(row, column)] - min) / span) * 2.0 - 1.0;
    }
}

fn build_cosine_knn(points: &[EmbeddingPoint], neighbors: usize) -> (Array2<u32>, Array2<f32>) {
    let rows = points.len();
    let mut indices = Array2::<u32>::zeros((rows, neighbors));
    let mut distances = Array2::<f32>::zeros((rows, neighbors));

    for (row, point) in points.iter().enumerate() {
        let mut scores = points
            .iter()
            .enumerate()
            .filter(|(other_row, _)| *other_row != row)
            .map(|(other_row, other)| {
                (
                    other_row,
                    cosine_distance(point.embedding.as_ref(), other.embedding.as_ref()),
                )
            })
            .collect::<Vec<_>>();

        scores.sort_by(|left, right| left.1.total_cmp(&right.1));
        for (column, (neighbor, distance)) in scores.into_iter().take(neighbors).enumerate() {
            indices[(row, column)] = neighbor as u32;
            distances[(row, column)] = distance;
        }
    }

    (indices, distances)
}

fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for (&lhs, &rhs) in left.iter().zip(right) {
        dot += lhs * rhs;
        left_norm += lhs * lhs;
        right_norm += rhs * rhs;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return 1.0;
    }

    let similarity = dot / (left_norm.sqrt() * right_norm.sqrt());
    (1.0 - similarity).clamp(0.0, 2.0)
}

fn cluster_feature_points(points: &[EmbeddingPoint]) -> Vec<i32> {
    if points.len() < DEFAULT_MIN_CLUSTER_SIZE {
        return vec![-1; points.len()];
    }

    let data = points
        .iter()
        .map(|point| point.embedding.as_ref().clone())
        .collect::<Vec<Vec<f32>>>();

    let hyper_params = HdbscanHyperParams::builder()
        .min_cluster_size(DEFAULT_MIN_CLUSTER_SIZE)
        .nn_algorithm(NnAlgorithm::BruteForce)
        .build();
    Hdbscan::new(&data, hyper_params)
        .cluster()
        .unwrap_or_else(|_| vec![-1; points.len()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FontFaceMetadata, StyleKind};
    use fontdb::ID;
    use image::GrayImage;

    fn fake_point(face_id: u64, embedding: Vec<f32>) -> EmbeddingPoint {
        EmbeddingPoint {
            face: FontFaceMetadata {
                face_id: ID::dummy(),
                family: format!("Family {face_id}"),
                style: "Normal".to_string(),
                style_kind: StyleKind::Normal,
                weight_value: 400,
                stretch_value: 5,
                post_script_name: format!("PS-{face_id}"),
                source_path: None,
                face_index: 0,
                monospaced: false,
            },
            image: Arc::new(GrayImage::new(224, 224)),
            embedding: Arc::new(embedding),
        }
    }

    fn test_config() -> HybridFeatureConfig {
        let dims = morphology_feature_count(4);
        HybridFeatureConfig {
            learned_scale: 0.4,
            morphology_scale: 2.4,
            projection_bins: 4,
            morphology_means: vec![0.5; dims],
            morphology_stds: vec![0.25; dims],
        }
    }

    #[test]
    fn cosine_knn_is_sorted() {
        let points = vec![
            fake_point(1, vec![1.0, 0.0]),
            fake_point(2, vec![0.9, 0.1]),
            fake_point(3, vec![-1.0, 0.0]),
        ];

        let (_, dists) = build_cosine_knn(&points, 2);
        assert!(dists[(0, 0)] <= dists[(0, 1)]);
    }

    #[test]
    fn hybrid_feature_normalization_is_deterministic() -> Result<()> {
        let config = test_config();
        let morphology = vec![0.6; morphology_feature_count(config.projection_bins)];

        let left = compose_hybrid_feature(&[1.0, 2.0, 3.0], &morphology, &config)?;
        let right = compose_hybrid_feature(&[1.0, 2.0, 3.0], &morphology, &config)?;

        assert_eq!(left, right);
        Ok(())
    }

    #[test]
    fn hdbscan_clusters_high_dim_features_without_umap() {
        let labels = cluster_feature_points(&[
            fake_point(1, vec![0.0, 0.0, 0.0, 0.0]),
            fake_point(2, vec![0.05, 0.02, 0.0, 0.0]),
            fake_point(3, vec![-0.03, 0.01, 0.0, 0.0]),
            fake_point(4, vec![0.02, -0.01, 0.0, 0.01]),
            fake_point(5, vec![-0.02, 0.03, 0.0, -0.01]),
            fake_point(6, vec![10.0, 10.0, 10.0, 10.0]),
            fake_point(7, vec![10.1, 10.0, 10.2, 10.1]),
            fake_point(8, vec![9.9, 10.1, 10.0, 10.2]),
            fake_point(9, vec![10.2, 9.9, 10.1, 10.0]),
            fake_point(10, vec![9.95, 10.05, 10.0, 9.9]),
        ]);

        assert_eq!(labels.len(), 10);
        assert!(labels.iter().any(|label| *label >= 0));
    }

    #[test]
    fn umap_reduction_returns_same_number_of_points() -> Result<()> {
        let points = vec![
            fake_point(1, vec![1.0, 0.0, 0.0]),
            fake_point(2, vec![0.9, 0.1, 0.0]),
            fake_point(3, vec![0.0, 1.0, 0.0]),
            fake_point(4, vec![0.0, 0.9, 0.1]),
            fake_point(5, vec![-1.0, 0.0, 0.0]),
            fake_point(6, vec![-0.9, 0.0, 0.1]),
        ];

        let reduced = reduce_embeddings(&points)?;
        assert_eq!(reduced.len(), points.len());
        Ok(())
    }
}
