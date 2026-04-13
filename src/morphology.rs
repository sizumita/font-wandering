use std::collections::VecDeque;

use image::GrayImage;

pub const INK_RATIO_INDEX: usize = 0;
pub const ASPECT_RATIO_INDEX: usize = 1;
pub const BBOX_FILL_RATIO_INDEX: usize = 2;
pub const CENTER_X_INDEX: usize = 3;
pub const CENTER_Y_INDEX: usize = 4;
pub const COMPONENT_COUNT_INDEX: usize = 5;
pub const HOLE_COUNT_INDEX: usize = 6;
pub const STROKE_MEAN_INDEX: usize = 7;
pub const STROKE_STD_INDEX: usize = 8;
pub const STROKE_MAX_INDEX: usize = 9;
pub const BASE_FEATURE_COUNT: usize = 10;

const FOREGROUND_THRESHOLD: u8 = 24;

#[derive(Clone, Copy, Debug)]
struct InkBounds {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

impl InkBounds {
    fn width(&self) -> usize {
        self.max_x - self.min_x + 1
    }

    fn height(&self) -> usize {
        self.max_y - self.min_y + 1
    }
}

pub fn morphology_feature_count(projection_bins: usize) -> usize {
    BASE_FEATURE_COUNT + projection_bins * 2
}

pub fn extract_morphology_features(image: &GrayImage, projection_bins: usize) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let pixel_count = (width * height).max(1);

    let mut weighted_ink = 0.0f32;
    let mut binary = vec![false; pixel_count];

    for (index, pixel) in image.pixels().enumerate() {
        let ink = pixel.0[0] as f32 / 255.0;
        weighted_ink += ink;
        binary[index] = pixel.0[0] >= FOREGROUND_THRESHOLD;
    }

    let ink_ratio = weighted_ink / pixel_count as f32;
    let bounds = ink_bounds(&binary, width, height);
    let mut features = Vec::with_capacity(morphology_feature_count(projection_bins));

    if let Some(bounds) = bounds {
        let bbox_area = (bounds.width() * bounds.height()).max(1) as f32;
        let bbox_weighted_ink = sum_region_ink(image, bounds);
        let (center_x, center_y) = weighted_centroid(image, bounds, weighted_ink);
        let components = connected_components(&binary, width, height, true) as f32;
        let holes = hole_count(&binary, width, height, bounds) as f32;
        let (stroke_mean, stroke_std, stroke_max) =
            stroke_width_stats(&binary, width, height, bounds);

        features.push(ink_ratio);
        features.push(bounds.width() as f32 / bounds.height().max(1) as f32);
        features.push((bbox_weighted_ink / bbox_area).clamp(0.0, 1.0));
        features.push(center_x);
        features.push(center_y);
        features.push((components / 12.0).clamp(0.0, 1.0));
        features.push((holes / 8.0).clamp(0.0, 1.0));
        features.push(stroke_mean);
        features.push(stroke_std);
        features.push(stroke_max);
        features.extend(projection_histogram(image, bounds, projection_bins, true));
        features.extend(projection_histogram(image, bounds, projection_bins, false));
    } else {
        features.resize(morphology_feature_count(projection_bins), 0.0);
    }

    features
}

fn ink_bounds(binary: &[bool], width: usize, height: usize) -> Option<InkBounds> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..height {
        for x in 0..width {
            if !binary[y * width + x] {
                continue;
            }
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            found = true;
        }
    }

    found.then_some(InkBounds {
        min_x,
        min_y,
        max_x,
        max_y,
    })
}

fn sum_region_ink(image: &GrayImage, bounds: InkBounds) -> f32 {
    let mut ink = 0.0;
    for y in bounds.min_y..=bounds.max_y {
        for x in bounds.min_x..=bounds.max_x {
            ink += image.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
        }
    }
    ink
}

fn weighted_centroid(image: &GrayImage, bounds: InkBounds, total_ink: f32) -> (f32, f32) {
    if total_ink <= f32::EPSILON {
        return (0.5, 0.5);
    }

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for y in bounds.min_y..=bounds.max_y {
        for x in bounds.min_x..=bounds.max_x {
            let ink = image.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
            if ink <= 0.0 {
                continue;
            }
            sum_x += x as f32 * ink;
            sum_y += y as f32 * ink;
        }
    }

    (
        (sum_x / total_ink) / image.width().max(1) as f32,
        (sum_y / total_ink) / image.height().max(1) as f32,
    )
}

fn connected_components(binary: &[bool], width: usize, height: usize, target: bool) -> usize {
    let mut visited = vec![false; binary.len()];
    let mut count = 0usize;

    for start in 0..binary.len() {
        if visited[start] || binary[start] != target {
            continue;
        }
        count += 1;
        flood_fill(binary, width, height, start, target, &mut visited, None);
    }

    count
}

fn hole_count(binary: &[bool], width: usize, height: usize, bounds: InkBounds) -> usize {
    let mut visited = vec![false; binary.len()];
    let mut holes = 0usize;

    for y in bounds.min_y..=bounds.max_y {
        for x in bounds.min_x..=bounds.max_x {
            let index = y * width + x;
            if visited[index] || binary[index] {
                continue;
            }

            let touches_border = flood_fill(
                binary,
                width,
                height,
                index,
                false,
                &mut visited,
                Some(bounds),
            );
            if !touches_border {
                holes += 1;
            }
        }
    }

    holes
}

fn flood_fill(
    binary: &[bool],
    width: usize,
    height: usize,
    start: usize,
    target: bool,
    visited: &mut [bool],
    restriction: Option<InkBounds>,
) -> bool {
    let mut queue = VecDeque::from([start]);
    visited[start] = true;
    let mut touches_border = false;

    while let Some(index) = queue.pop_front() {
        let x = index % width;
        let y = index / width;

        if let Some(bounds) = restriction {
            if x == bounds.min_x || x == bounds.max_x || y == bounds.min_y || y == bounds.max_y {
                touches_border = true;
            }
        }

        for (nx, ny) in neighbors(x, y, width, height) {
            if let Some(bounds) = restriction {
                if nx < bounds.min_x || nx > bounds.max_x || ny < bounds.min_y || ny > bounds.max_y
                {
                    continue;
                }
            }

            let neighbor = ny * width + nx;
            if visited[neighbor] || binary[neighbor] != target {
                continue;
            }
            visited[neighbor] = true;
            queue.push_back(neighbor);
        }
    }

    touches_border
}

fn stroke_width_stats(
    binary: &[bool],
    width: usize,
    height: usize,
    bounds: InkBounds,
) -> (f32, f32, f32) {
    let distances = cityblock_distance_transform(binary, width, height);
    let mut samples = Vec::new();
    let normalizer = bounds.width().max(bounds.height()).max(1) as f32;

    for y in bounds.min_y..=bounds.max_y {
        for x in bounds.min_x..=bounds.max_x {
            let index = y * width + x;
            if !binary[index] {
                continue;
            }
            samples.push(distances[index] as f32 / normalizer);
        }
    }

    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let variance = samples
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / samples.len() as f32;
    let max = samples
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0);

    (mean, variance.sqrt(), max)
}

fn cityblock_distance_transform(binary: &[bool], width: usize, height: usize) -> Vec<u16> {
    let inf = u16::MAX / 4;
    let mut distances = vec![inf; binary.len()];

    for (index, value) in binary.iter().enumerate() {
        if !value {
            distances[index] = 0;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let mut best = distances[index];
            if x > 0 {
                best = best.min(distances[index - 1].saturating_add(1));
            }
            if y > 0 {
                best = best.min(distances[index - width].saturating_add(1));
            }
            distances[index] = best;
        }
    }

    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let index = y * width + x;
            let mut best = distances[index];
            if x + 1 < width {
                best = best.min(distances[index + 1].saturating_add(1));
            }
            if y + 1 < height {
                best = best.min(distances[index + width].saturating_add(1));
            }
            distances[index] = best;
        }
    }

    distances
}

fn projection_histogram(
    image: &GrayImage,
    bounds: InkBounds,
    bins: usize,
    horizontal: bool,
) -> Vec<f32> {
    if bins == 0 {
        return Vec::new();
    }

    let span = if horizontal {
        bounds.height()
    } else {
        bounds.width()
    };
    let mut histogram = vec![0.0f32; bins];
    let mut total = 0.0f32;

    if horizontal {
        for y in bounds.min_y..=bounds.max_y {
            let mut line_sum = 0.0;
            for x in bounds.min_x..=bounds.max_x {
                line_sum += image.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
            }
            let bucket = ((y - bounds.min_y) * bins / span.max(1)).min(bins - 1);
            histogram[bucket] += line_sum;
            total += line_sum;
        }
    } else {
        for x in bounds.min_x..=bounds.max_x {
            let mut line_sum = 0.0;
            for y in bounds.min_y..=bounds.max_y {
                line_sum += image.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
            }
            let bucket = ((x - bounds.min_x) * bins / span.max(1)).min(bins - 1);
            histogram[bucket] += line_sum;
            total += line_sum;
        }
    }

    if total > f32::EPSILON {
        for value in &mut histogram {
            *value /= total;
        }
    }

    histogram
}

fn neighbors(x: usize, y: usize, width: usize, height: usize) -> [(usize, usize); 4] {
    [
        (x.saturating_sub(1), y),
        ((x + 1).min(width - 1), y),
        (x, y.saturating_sub(1)),
        (x, (y + 1).min(height - 1)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn rectangle_image(
        width: u32,
        height: u32,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
    ) -> GrayImage {
        let mut image = GrayImage::from_pixel(width, height, Luma([0]));
        for y in top..bottom {
            for x in left..right {
                image.put_pixel(x, y, Luma([255]));
            }
        }
        image
    }

    #[test]
    fn thicker_shapes_increase_ink_and_stroke_proxy() {
        let thin = rectangle_image(64, 64, 28, 10, 36, 54);
        let thick = rectangle_image(64, 64, 20, 10, 44, 54);

        let thin_features = extract_morphology_features(&thin, 4);
        let thick_features = extract_morphology_features(&thick, 4);

        assert!(thick_features[INK_RATIO_INDEX] > thin_features[INK_RATIO_INDEX]);
        assert!(thick_features[STROKE_MEAN_INDEX] > thin_features[STROKE_MEAN_INDEX]);
        assert!(thick_features[STROKE_MAX_INDEX] > thin_features[STROKE_MAX_INDEX]);
    }

    #[test]
    fn enclosed_regions_raise_hole_count() {
        let mut ring = GrayImage::from_pixel(64, 64, Luma([0]));
        for y in 12..52 {
            for x in 12..52 {
                if !(24..40).contains(&x) || !(24..40).contains(&y) {
                    ring.put_pixel(x, y, Luma([255]));
                }
            }
        }

        let features = extract_morphology_features(&ring, 4);
        assert!(features[HOLE_COUNT_INDEX] > 0.0);
    }
}
