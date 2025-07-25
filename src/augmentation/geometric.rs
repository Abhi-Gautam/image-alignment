use crate::augmentation::base::*;
use crate::pipeline::{AugmentedImage, GroundTruth, ImageAugmentation, Transform, TransformType};
use crate::Result;
use opencv::core::{Mat, Point2f, Size};
use opencv::imgproc;
use opencv::prelude::*;
use rand::Rng;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Barrel distortion augmentation (lens distortion effect)
pub struct BarrelDistortionAugmentation {
    base: AugmentationBase,
    k1_range: (f64, f64), // Radial distortion coefficient range
}

impl BarrelDistortionAugmentation {
    pub fn new(k1_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            k1_range,
        }
    }
}

impl ImageAugmentation for BarrelDistortionAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let k1 = base.random_in_range(self.k1_range.0, self.k1_range.1);

        let width = image.cols();
        let height = image.rows();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;

        // Create distortion map
        let mut map_x = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;
        let mut map_y = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;

        let max_radius = (center_x * center_x + center_y * center_y).sqrt();

        for y in 0..height {
            for x in 0..width {
                let norm_x = (x as f64 - center_x) / max_radius;
                let norm_y = (y as f64 - center_y) / max_radius;
                let r2 = norm_x * norm_x + norm_y * norm_y;

                // Radial distortion formula: r' = r * (1 + k1 * r^2)
                let distortion_factor = 1.0 + k1 * r2;

                let src_x = center_x + norm_x * max_radius * distortion_factor;
                let src_y = center_y + norm_y * max_radius * distortion_factor;

                *map_x.at_2d_mut::<f32>(y, x)? = src_x as f32;
                *map_y.at_2d_mut::<f32>(y, x)? = src_y as f32;
            }
        }

        let mut output = Mat::default();
        imgproc::remap(
            image,
            &mut output,
            &map_x,
            &map_y,
            imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::all(0.0),
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
            augmentations_applied: vec!["barrel_distortion".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("k1".to_string(), json!(k1));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(Transform {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            transform_type: TransformType::Combined,
        })
    }

    fn description(&self) -> String {
        format!("Barrel distortion with k1 in range {:?}", self.k1_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("barrel_distortion"));
        params.insert("k1_range".to_string(), json!(self.k1_range));
        params
    }
}

/// Pincushion distortion augmentation (opposite of barrel distortion)
pub struct PincushionDistortionAugmentation {
    base: AugmentationBase,
    k1_range: (f64, f64), // Radial distortion coefficient range (negative for pincushion)
}

impl PincushionDistortionAugmentation {
    pub fn new(k1_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            k1_range,
        }
    }
}

impl ImageAugmentation for PincushionDistortionAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let k1 = -base.random_in_range(self.k1_range.0.abs(), self.k1_range.1.abs()); // Negative for pincushion

        let width = image.cols();
        let height = image.rows();
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;

        // Create distortion map (same as barrel but with negative k1)
        let mut map_x = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;
        let mut map_y = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;

        let max_radius = (center_x * center_x + center_y * center_y).sqrt();

        for y in 0..height {
            for x in 0..width {
                let norm_x = (x as f64 - center_x) / max_radius;
                let norm_y = (y as f64 - center_y) / max_radius;
                let r2 = norm_x * norm_x + norm_y * norm_y;

                let distortion_factor = 1.0 + k1 * r2;

                let src_x = center_x + norm_x * max_radius * distortion_factor;
                let src_y = center_y + norm_y * max_radius * distortion_factor;

                *map_x.at_2d_mut::<f32>(y, x)? = src_x as f32;
                *map_y.at_2d_mut::<f32>(y, x)? = src_y as f32;
            }
        }

        let mut output = Mat::default();
        imgproc::remap(
            image,
            &mut output,
            &map_x,
            &map_y,
            imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::all(0.0),
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
            augmentations_applied: vec!["pincushion_distortion".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("k1".to_string(), json!(k1));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(Transform {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            transform_type: TransformType::Combined,
        })
    }

    fn description(&self) -> String {
        format!("Pincushion distortion with k1 in range {:?}", self.k1_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("pincushion_distortion"));
        params.insert("k1_range".to_string(), json!(self.k1_range));
        params
    }
}

/// Perspective warp augmentation (simulates viewing angle changes)
pub struct PerspectiveWarpAugmentation {
    base: AugmentationBase,
    warp_strength_range: (f64, f64), // Controls how much perspective is applied
}

impl PerspectiveWarpAugmentation {
    pub fn new(warp_strength_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            warp_strength_range,
        }
    }
}

impl ImageAugmentation for PerspectiveWarpAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let warp_strength =
            base.random_in_range(self.warp_strength_range.0, self.warp_strength_range.1);

        let width = image.cols() as f32;
        let height = image.rows() as f32;

        // Define source points (image corners)
        let src_points = vec![
            Point2f::new(0.0, 0.0),
            Point2f::new(width, 0.0),
            Point2f::new(width, height),
            Point2f::new(0.0, height),
        ];

        // Define destination points with random perturbation
        let max_offset = (warp_strength as f32) * width.min(height) * 0.1;
        let dst_points = vec![
            Point2f::new(
                base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
                base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
            ),
            Point2f::new(
                width + base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
                base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
            ),
            Point2f::new(
                width + base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
                height + base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
            ),
            Point2f::new(
                base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
                height + base.random_in_range(-max_offset as f64, max_offset as f64) as f32,
            ),
        ];

        // Calculate perspective transform matrix
        let src_mat = opencv::core::Mat::from_slice(&src_points)?;
        let dst_mat = opencv::core::Mat::from_slice(&dst_points)?;
        let transform_matrix =
            imgproc::get_perspective_transform(&src_mat, &dst_mat, opencv::core::DECOMP_LU)?;

        let mut output = Mat::default();
        imgproc::warp_perspective(
            image,
            &mut output,
            &transform_matrix,
            Size::new(width as i32, height as i32),
            imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::all(0.0),
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
            augmentations_applied: vec!["perspective_warp".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("warp_strength".to_string(), json!(warp_strength));
                params.insert(
                    "src_points".to_string(),
                    json!(src_points.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()),
                );
                params.insert(
                    "dst_points".to_string(),
                    json!(dst_points.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()),
                );
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(Transform {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            transform_type: TransformType::Perspective,
        })
    }

    fn description(&self) -> String {
        format!(
            "Perspective warp with strength in range {:?}",
            self.warp_strength_range
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("perspective_warp"));
        params.insert(
            "warp_strength_range".to_string(),
            json!(self.warp_strength_range),
        );
        params
    }
}

/// Elastic deformation augmentation (simulates material deformation)
pub struct ElasticDeformationAugmentation {
    base: AugmentationBase,
    alpha_range: (f64, f64), // Deformation strength
    sigma_range: (f64, f64), // Smoothness of deformation
}

impl ElasticDeformationAugmentation {
    pub fn new(alpha_range: (f64, f64), sigma_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            alpha_range,
            sigma_range,
        }
    }
}

impl ImageAugmentation for ElasticDeformationAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let alpha = base.random_in_range(self.alpha_range.0, self.alpha_range.1);
        let sigma = base.random_in_range(self.sigma_range.0, self.sigma_range.1);

        let width = image.cols();
        let height = image.rows();

        // Generate random displacement fields
        let mut dx_field = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;
        let mut dy_field = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;

        // Fill with random noise
        for y in 0..height {
            for x in 0..width {
                *dx_field.at_2d_mut::<f32>(y, x)? =
                    (Rng::gen::<f32>(&mut base.rng) - 0.5) * 2.0 * alpha as f32;
                *dy_field.at_2d_mut::<f32>(y, x)? =
                    (Rng::gen::<f32>(&mut base.rng) - 0.5) * 2.0 * alpha as f32;
            }
        }

        // Smooth the displacement fields with Gaussian filter
        let ksize = Size::new(0, 0);
        let mut smooth_dx = Mat::default();
        let mut smooth_dy = Mat::default();

        opencv::imgproc::gaussian_blur(
            &dx_field,
            &mut smooth_dx,
            ksize,
            sigma,
            sigma,
            opencv::core::BORDER_DEFAULT,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        opencv::imgproc::gaussian_blur(
            &dy_field,
            &mut smooth_dy,
            ksize,
            sigma,
            sigma,
            opencv::core::BORDER_DEFAULT,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Create mapping arrays
        let mut map_x = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;
        let mut map_y = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;

        for y in 0..height {
            for x in 0..width {
                *map_x.at_2d_mut::<f32>(y, x)? = x as f32 + *smooth_dx.at_2d::<f32>(y, x)?;
                *map_y.at_2d_mut::<f32>(y, x)? = y as f32 + *smooth_dy.at_2d::<f32>(y, x)?;
            }
        }

        let mut output = Mat::default();
        imgproc::remap(
            image,
            &mut output,
            &map_x,
            &map_y,
            imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::all(0.0),
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
            augmentations_applied: vec!["elastic_deformation".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("alpha".to_string(), json!(alpha));
                params.insert("sigma".to_string(), json!(sigma));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Complex elastic deformation is not easily invertible
    }

    fn description(&self) -> String {
        format!(
            "Elastic deformation with alpha {:?} and sigma {:?}",
            self.alpha_range, self.sigma_range
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("elastic_deformation"));
        params.insert("alpha_range".to_string(), json!(self.alpha_range));
        params.insert("sigma_range".to_string(), json!(self.sigma_range));
        params
    }
}
