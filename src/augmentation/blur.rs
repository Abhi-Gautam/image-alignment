use crate::augmentation::base::*;
use crate::pipeline::{AugmentedImage, GroundTruth, ImageAugmentation, Transform};
use crate::Result;
use opencv::core::{Mat, Point, Size};
use opencv::imgproc;
use opencv::prelude::*;
use rand::Rng;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Gaussian blur augmentation
pub struct GaussianBlurAugmentation {
    base: AugmentationBase,
    sigma_range: (f64, f64), // Standard deviation range for blur kernel
}

impl GaussianBlurAugmentation {
    pub fn new(sigma_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            sigma_range,
        }
    }
}

impl ImageAugmentation for GaussianBlurAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let sigma = base.random_in_range(self.sigma_range.0, self.sigma_range.1);

        let mut output = Mat::default();
        let ksize = Size::new(0, 0); // Auto-calculate kernel size from sigma

        imgproc::gaussian_blur(
            image,
            &mut output,
            ksize,
            sigma,
            sigma,
            opencv::core::BORDER_DEFAULT,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let ground_truth = GroundTruth {
            expected_location: crate::pipeline::SerializableRect {
                x: 0,
                y: 0,
                width: image.cols(),
                height: image.rows(),
            },
            transformation: crate::pipeline::TransformParams {
                translation: (0.0, 0.0),
                rotation_degrees: 0.0,
                scale: 1.0,
                skew: None,
            },
            metadata: HashMap::new(),
        };

        Ok(AugmentedImage {
            image: output,
            original: image.clone(),
            ground_truth,
            augmentations_applied: vec!["gaussian_blur".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("sigma".to_string(), json!(sigma));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Blur is not easily invertible
    }

    fn description(&self) -> String {
        format!("Gaussian blur with sigma in range {:?}", self.sigma_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("gaussian_blur"));
        params.insert("sigma_range".to_string(), json!(self.sigma_range));
        params
    }
}

/// Motion blur augmentation (simulates camera shake or object motion)
pub struct MotionBlurAugmentation {
    base: AugmentationBase,
    length_range: (i32, i32), // Motion blur length in pixels
    angle_range: (f64, f64),  // Motion blur angle in degrees
}

impl MotionBlurAugmentation {
    pub fn new(length_range: (i32, i32), angle_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            length_range,
            angle_range,
        }
    }
}

impl ImageAugmentation for MotionBlurAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let length = Rng::gen_range(&mut base.rng, self.length_range.0..=self.length_range.1);
        let angle = base.random_in_range(self.angle_range.0, self.angle_range.1);

        // Create motion blur kernel
        let kernel = create_motion_blur_kernel(length, angle)?;

        let mut output = Mat::default();
        imgproc::filter_2d(
            image,
            &mut output,
            -1,
            &kernel,
            Point::new(-1, -1),
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        let ground_truth = GroundTruth {
            expected_location: crate::pipeline::SerializableRect {
                x: 0,
                y: 0,
                width: image.cols(),
                height: image.rows(),
            },
            transformation: crate::pipeline::TransformParams {
                translation: (0.0, 0.0),
                rotation_degrees: 0.0,
                scale: 1.0,
                skew: None,
            },
            metadata: HashMap::new(),
        };

        Ok(AugmentedImage {
            image: output,
            original: image.clone(),
            ground_truth,
            augmentations_applied: vec!["motion_blur".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("length".to_string(), json!(length));
                params.insert("angle".to_string(), json!(angle));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Motion blur is not easily invertible
    }

    fn description(&self) -> String {
        format!(
            "Motion blur with length {:?} and angle {:?}",
            self.length_range, self.angle_range
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("motion_blur"));
        params.insert("length_range".to_string(), json!(self.length_range));
        params.insert("angle_range".to_string(), json!(self.angle_range));
        params
    }
}

/// Defocus blur augmentation (simulates out-of-focus images)
pub struct DefocusBlurAugmentation {
    base: AugmentationBase,
    radius_range: (i32, i32), // Defocus radius range
}

impl DefocusBlurAugmentation {
    pub fn new(radius_range: (i32, i32)) -> Self {
        Self {
            base: AugmentationBase::default(),
            radius_range,
        }
    }
}

impl ImageAugmentation for DefocusBlurAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let radius = Rng::gen_range(&mut base.rng, self.radius_range.0..=self.radius_range.1);

        // Create disk-shaped kernel for defocus blur
        let kernel = create_disk_kernel(radius)?;

        let mut output = Mat::default();
        imgproc::filter_2d(
            image,
            &mut output,
            -1,
            &kernel,
            Point::new(-1, -1),
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        let ground_truth = GroundTruth {
            expected_location: crate::pipeline::SerializableRect {
                x: 0,
                y: 0,
                width: image.cols(),
                height: image.rows(),
            },
            transformation: crate::pipeline::TransformParams {
                translation: (0.0, 0.0),
                rotation_degrees: 0.0,
                scale: 1.0,
                skew: None,
            },
            metadata: HashMap::new(),
        };

        Ok(AugmentedImage {
            image: output,
            original: image.clone(),
            ground_truth,
            augmentations_applied: vec!["defocus_blur".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("radius".to_string(), json!(radius));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Defocus blur is not easily invertible
    }

    fn description(&self) -> String {
        format!("Defocus blur with radius in range {:?}", self.radius_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("defocus_blur"));
        params.insert("radius_range".to_string(), json!(self.radius_range));
        params
    }
}

/// Box blur augmentation (simple averaging filter)
pub struct BoxBlurAugmentation {
    base: AugmentationBase,
    size_range: (i32, i32), // Kernel size range (must be odd)
}

impl BoxBlurAugmentation {
    pub fn new(size_range: (i32, i32)) -> Self {
        Self {
            base: AugmentationBase::default(),
            size_range,
        }
    }
}

impl ImageAugmentation for BoxBlurAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let mut size = Rng::gen_range(&mut base.rng, self.size_range.0..=self.size_range.1);

        // Ensure size is odd
        if size % 2 == 0 {
            size += 1;
        }

        let mut output = Mat::default();
        let ksize = Size::new(size, size);

        imgproc::box_filter(
            image,
            &mut output,
            -1,
            ksize,
            Point::new(-1, -1),
            true, // Normalize
            opencv::core::BORDER_DEFAULT,
        )?;

        let ground_truth = GroundTruth {
            expected_location: crate::pipeline::SerializableRect {
                x: 0,
                y: 0,
                width: image.cols(),
                height: image.rows(),
            },
            transformation: crate::pipeline::TransformParams {
                translation: (0.0, 0.0),
                rotation_degrees: 0.0,
                scale: 1.0,
                skew: None,
            },
            metadata: HashMap::new(),
        };

        Ok(AugmentedImage {
            image: output,
            original: image.clone(),
            ground_truth,
            augmentations_applied: vec!["box_blur".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("size".to_string(), json!(size));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Box blur is not easily invertible
    }

    fn description(&self) -> String {
        format!("Box blur with size in range {:?}", self.size_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("box_blur"));
        params.insert("size_range".to_string(), json!(self.size_range));
        params
    }
}

// Helper function to create motion blur kernel
fn create_motion_blur_kernel(length: i32, angle: f64) -> Result<Mat> {
    let angle_rad = angle.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Calculate kernel size needed to contain the motion line
    let half_len = length as f64 / 2.0;
    let width = (half_len * cos_a.abs()).ceil() as i32 * 2 + 1;
    let height = (half_len * sin_a.abs()).ceil() as i32 * 2 + 1;

    let mut kernel = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;

    let center_x = width / 2;
    let center_y = height / 2;

    // Draw line in kernel
    let mut count = 0;
    for t in 0..length {
        let x = center_x + (((t - length / 2) as f64) * cos_a) as i32;
        let y = center_y + (((t - length / 2) as f64) * sin_a) as i32;

        if x >= 0 && x < width && y >= 0 && y < height {
            *kernel.at_2d_mut::<f32>(y, x)? = 1.0;
            count += 1;
        }
    }

    // Normalize kernel
    if count > 0 {
        let normalization_factor = 1.0 / count as f32;
        let mut normalized = Mat::default();
        opencv::core::multiply(
            &kernel,
            &opencv::core::Scalar::all(normalization_factor as f64),
            &mut normalized,
            1.0,
            -1,
        )?;
        kernel = normalized;
    }

    Ok(kernel)
}

// Helper function to create disk-shaped kernel for defocus blur
fn create_disk_kernel(radius: i32) -> Result<Mat> {
    let size = radius * 2 + 1;
    let mut kernel = Mat::zeros(size, size, opencv::core::CV_32F)?.to_mat()?;

    let center = radius;
    let mut count = 0;

    for y in 0..size {
        for x in 0..size {
            let dx = x - center;
            let dy = y - center;
            let distance = ((dx * dx + dy * dy) as f64).sqrt();

            if distance <= radius as f64 {
                *kernel.at_2d_mut::<f32>(y, x)? = 1.0;
                count += 1;
            }
        }
    }

    // Normalize kernel
    if count > 0 {
        let normalization_factor = 1.0 / count as f32;
        let mut normalized = Mat::default();
        opencv::core::multiply(
            &kernel,
            &opencv::core::Scalar::all(normalization_factor as f64),
            &mut normalized,
            1.0,
            -1,
        )?;
        kernel = normalized;
    }

    Ok(kernel)
}
