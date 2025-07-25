use crate::pipeline::{Transform, TransformType};
use crate::Result;
use opencv::core::Mat;
use opencv::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Base implementation helpers for augmentations
#[derive(Clone)]
pub struct AugmentationBase {
    pub rng: StdRng,
    pub preserve_original: bool,
}

impl Default for AugmentationBase {
    fn default() -> Self {
        Self {
            rng: StdRng::from_entropy(),
            preserve_original: true,
        }
    }
}

impl AugmentationBase {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            preserve_original: true,
        }
    }

    pub fn random_in_range(&mut self, min: f64, max: f64) -> f64 {
        self.rng.gen_range(min..=max)
    }

    pub fn random_bool(&mut self, probability: f64) -> bool {
        self.rng.gen_bool(probability)
    }
}

/// Configuration for augmentation ranges
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    pub intensity_range: (f64, f64),
    pub probability: f64,
    pub preserve_edges: bool,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            intensity_range: (0.0, 1.0),
            probability: 1.0,
            preserve_edges: true,
        }
    }
}

/// Helper trait for converting Mat between types
pub trait MatConversion {
    fn to_f32(&self) -> Result<Mat>;
    fn to_u8(&self) -> Result<Mat>;
    fn normalize_to_u8(&self) -> Result<Mat>;
}

impl MatConversion for Mat {
    fn to_f32(&self) -> Result<Mat> {
        let mut output = Mat::default();
        self.convert_to(&mut output, opencv::core::CV_32F, 1.0, 0.0)?;
        Ok(output)
    }

    fn to_u8(&self) -> Result<Mat> {
        let mut output = Mat::default();
        self.convert_to(&mut output, opencv::core::CV_8U, 1.0, 0.0)?;
        Ok(output)
    }

    fn normalize_to_u8(&self) -> Result<Mat> {
        let mut normalized = Mat::default();
        opencv::core::normalize(
            self,
            &mut normalized,
            0.0,
            255.0,
            opencv::core::NORM_MINMAX,
            opencv::core::CV_8U,
            &opencv::core::no_array(),
        )?;
        Ok(normalized)
    }
}

/// Create identity transform
pub fn identity_transform() -> Transform {
    Transform {
        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        transform_type: TransformType::Translation,
    }
}

/// Create translation transform
pub fn translation_transform(dx: f64, dy: f64) -> Transform {
    Transform {
        matrix: [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
        transform_type: TransformType::Translation,
    }
}

/// Create rotation transform (in radians)
pub fn rotation_transform(angle: f64, cx: f64, cy: f64) -> Transform {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    Transform {
        matrix: [
            [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
            [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
            [0.0, 0.0, 1.0],
        ],
        transform_type: TransformType::Rotation,
    }
}

/// Create scale transform
pub fn scale_transform(sx: f64, sy: f64, cx: f64, cy: f64) -> Transform {
    Transform {
        matrix: [
            [sx, 0.0, cx - cx * sx],
            [0.0, sy, cy - cy * sy],
            [0.0, 0.0, 1.0],
        ],
        transform_type: TransformType::Scale,
    }
}

/// Combine two transforms
pub fn combine_transforms(t1: &Transform, t2: &Transform) -> Transform {
    let mut result = [[0.0; 3]; 3];

    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += t1.matrix[i][k] * t2.matrix[k][j];
            }
        }
    }

    Transform {
        matrix: result,
        transform_type: TransformType::Combined,
    }
}
