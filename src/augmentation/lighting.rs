use crate::augmentation::base::*;
use crate::pipeline::{AugmentedImage, GroundTruth, ImageAugmentation, Transform};
use crate::Result;
use opencv::core::{Mat, Scalar};
use opencv::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Brightness adjustment augmentation
pub struct BrightnessAugmentation {
    base: AugmentationBase,
    brightness_range: (f64, f64), // (-100, 100) typical range
}

impl BrightnessAugmentation {
    pub fn new(brightness_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            brightness_range,
        }
    }
}

impl ImageAugmentation for BrightnessAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let brightness = base.random_in_range(self.brightness_range.0, self.brightness_range.1);

        let mut output = Mat::default();
        image.convert_to(&mut output, -1, 1.0, brightness)?;

        // Clamp values to valid range
        let mut clamped = Mat::default();
        opencv::core::min(&output, &Scalar::all(255.0), &mut clamped)?;
        opencv::core::max(&clamped, &Scalar::all(0.0), &mut output)?;

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
            augmentations_applied: vec!["brightness".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("brightness".to_string(), json!(brightness));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(identity_transform())
    }

    fn description(&self) -> String {
        format!("Brightness adjustment in range {:?}", self.brightness_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("brightness"));
        params.insert("range".to_string(), json!(self.brightness_range));
        params
    }
}

/// Contrast adjustment augmentation
pub struct ContrastAugmentation {
    base: AugmentationBase,
    contrast_range: (f64, f64), // (0.5, 2.0) typical range
}

impl ContrastAugmentation {
    pub fn new(contrast_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            contrast_range,
        }
    }
}

impl ImageAugmentation for ContrastAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let contrast = base.random_in_range(self.contrast_range.0, self.contrast_range.1);

        // Calculate mean for contrast adjustment
        let mean = opencv::core::mean(image, &opencv::core::no_array())?;
        let mean_value = mean[0];

        let mut output = Mat::default();
        image.convert_to(&mut output, -1, contrast, mean_value * (1.0 - contrast))?;

        // Clamp values
        let mut clamped = Mat::default();
        opencv::core::min(&output, &Scalar::all(255.0), &mut clamped)?;
        opencv::core::max(&clamped, &Scalar::all(0.0), &mut output)?;

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
            augmentations_applied: vec!["contrast".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("contrast".to_string(), json!(contrast));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(identity_transform())
    }

    fn description(&self) -> String {
        format!("Contrast adjustment in range {:?}", self.contrast_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("contrast"));
        params.insert("range".to_string(), json!(self.contrast_range));
        params
    }
}

/// Gamma correction augmentation
pub struct GammaAugmentation {
    base: AugmentationBase,
    gamma_range: (f64, f64), // (0.5, 2.0) typical range
}

impl GammaAugmentation {
    pub fn new(gamma_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            gamma_range,
        }
    }
}

impl ImageAugmentation for GammaAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let gamma = base.random_in_range(self.gamma_range.0, self.gamma_range.1);

        // Create lookup table for gamma correction
        let mut lookup_table = vec![0u8; 256];
        #[allow(clippy::needless_range_loop)]
        for i in 0..256 {
            lookup_table[i] = ((i as f64 / 255.0).powf(1.0 / gamma) * 255.0) as u8;
        }

        // Apply lookup table
        let mut output = Mat::default();
        let lut_mat = Mat::from_slice(&lookup_table)?;
        opencv::core::lut(image, &lut_mat, &mut output)?;

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
            augmentations_applied: vec!["gamma".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("gamma".to_string(), json!(gamma));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(identity_transform())
    }

    fn description(&self) -> String {
        format!("Gamma correction in range {:?}", self.gamma_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("gamma"));
        params.insert("range".to_string(), json!(self.gamma_range));
        params
    }
}

/// Local illumination variation (simulates uneven lighting)
pub struct LocalIlluminationAugmentation {
    base: AugmentationBase,
    intensity_range: (f64, f64),
    gradient_type: GradientType,
}

#[derive(Clone, Debug)]
pub enum GradientType {
    Linear,
    Radial,
    Corner,
}

impl LocalIlluminationAugmentation {
    pub fn new(intensity_range: (f64, f64), gradient_type: GradientType) -> Self {
        Self {
            base: AugmentationBase::default(),
            intensity_range,
            gradient_type,
        }
    }
}

impl ImageAugmentation for LocalIlluminationAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let intensity = base.random_in_range(self.intensity_range.0, self.intensity_range.1);

        let height = image.rows();
        let width = image.cols();

        // Create gradient mask
        let mut mask = Mat::zeros(height, width, opencv::core::CV_32F)?.to_mat()?;

        match self.gradient_type {
            GradientType::Linear => {
                // Linear gradient from top to bottom
                for y in 0..height {
                    let value = (y as f32 / height as f32) * intensity as f32;
                    for x in 0..width {
                        *mask.at_2d_mut::<f32>(y, x)? = value;
                    }
                }
            }
            GradientType::Radial => {
                // Radial gradient from center
                let cx = width as f32 / 2.0;
                let cy = height as f32 / 2.0;
                let max_dist = (cx * cx + cy * cy).sqrt();

                for y in 0..height {
                    for x in 0..width {
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let value = (dist / max_dist) * intensity as f32;
                        *mask.at_2d_mut::<f32>(y, x)? = value;
                    }
                }
            }
            GradientType::Corner => {
                // Gradient from top-left corner
                let max_dist = ((width * width + height * height) as f32).sqrt();

                for y in 0..height {
                    for x in 0..width {
                        let dist = ((x * x + y * y) as f32).sqrt();
                        let value = (dist / max_dist) * intensity as f32;
                        *mask.at_2d_mut::<f32>(y, x)? = value;
                    }
                }
            }
        }

        // Apply illumination variation
        let image_f32 = image.to_f32()?;
        let mut output = Mat::default();
        opencv::core::add(
            &image_f32,
            &mask,
            &mut output,
            &opencv::core::no_array(),
            -1,
        )?;
        let output = output.normalize_to_u8()?;

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
            augmentations_applied: vec!["local_illumination".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("intensity".to_string(), json!(intensity));
                params.insert(
                    "gradient_type".to_string(),
                    json!(format!("{:?}", self.gradient_type)),
                );
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(identity_transform())
    }

    fn description(&self) -> String {
        format!(
            "Local illumination variation ({:?}) in range {:?}",
            self.gradient_type, self.intensity_range
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("local_illumination"));
        params.insert("range".to_string(), json!(self.intensity_range));
        params.insert(
            "gradient_type".to_string(),
            json!(format!("{:?}", self.gradient_type)),
        );
        params
    }
}
