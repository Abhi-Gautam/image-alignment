use crate::augmentation::base::*;
use crate::config::AugmentationConfig;
use crate::pipeline::{AugmentedImage, ImageAugmentation, Transform};
use crate::utils::{grayimage_to_mat, image_to_grayimage};
use crate::Result;
use opencv::core::Mat;
use opencv::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Composite augmentation that applies multiple augmentations in sequence
pub struct CompositeAugmentation {
    augmentations: Vec<Box<dyn ImageAugmentation>>,
    apply_probability: f64,
}

impl Default for CompositeAugmentation {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeAugmentation {
    pub fn new() -> Self {
        Self {
            augmentations: Vec::new(),
            apply_probability: 1.0,
        }
    }

    pub fn add_augmentation(mut self, augmentation: Box<dyn ImageAugmentation>) -> Self {
        self.augmentations.push(augmentation);
        self
    }

    pub fn with_probability(mut self, probability: f64) -> Self {
        self.apply_probability = probability.clamp(0.0, 1.0);
        self
    }
}

impl ImageAugmentation for CompositeAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut current_image = image.clone();
        let mut all_augmentations = Vec::new();
        let mut all_params = HashMap::new();
        let mut base = AugmentationBase::default();

        // Apply each augmentation with probability
        for (i, augmentation) in self.augmentations.iter().enumerate() {
            if base.random_bool(self.apply_probability) {
                let augmented = augmentation.apply(&current_image)?;
                current_image = augmented.image;

                // Collect applied augmentations
                all_augmentations.extend(augmented.augmentations_applied);
                for (key, value) in augmented.augmentation_params {
                    all_params.insert(format!("step_{}_{}", i, key), value);
                }
            }
        }

        // Use the ground truth from the last applied augmentation
        // In practice, you might want to compose the transformations
        let ground_truth = crate::pipeline::GroundTruth {
            expected_location: crate::pipeline::SerializableRect {
                x: 0,
                y: 0,
                width: current_image.cols(),
                height: current_image.rows(),
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
            image: current_image,
            original: image.clone(),
            ground_truth,
            augmentations_applied: all_augmentations,
            augmentation_params: all_params,
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        // For composite transforms, we'd need to compose inverse transforms
        // This is complex and often not possible, so return None
        None
    }

    fn description(&self) -> String {
        format!(
            "Composite augmentation with {} steps",
            self.augmentations.len()
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("composite"));
        params.insert(
            "num_augmentations".to_string(),
            json!(self.augmentations.len()),
        );
        params.insert(
            "apply_probability".to_string(),
            json!(self.apply_probability),
        );
        params
    }
}

/// Realistic SEM augmentation pipeline that combines effects commonly seen in SEM imaging
pub struct RealisticSEMAugmentation {
    base: AugmentationBase,
    intensity: f64, // Overall intensity of effects (0.0 to 1.0)
    config: AugmentationConfig,
}

impl RealisticSEMAugmentation {
    pub fn new(intensity: f64) -> Self {
        Self {
            base: AugmentationBase::default(),
            intensity: intensity.clamp(0.0, 1.0),
            config: AugmentationConfig::default(),
        }
    }

    pub fn with_config(mut self, config: AugmentationConfig) -> Self {
        self.config = config;
        self
    }
}

impl ImageAugmentation for RealisticSEMAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        use crate::augmentation::*;

        // Create a composite augmentation with realistic SEM effects
        let mut composite = CompositeAugmentation::new();

        // Scale effect intensities based on overall intensity and config
        let light_intensity = self.intensity * 0.3;
        let noise_intensity = self.intensity * 0.2;
        let blur_intensity = self.intensity * 0.1;
        let distortion_intensity = self.intensity * 0.1;

        // Add lighting variations (common in SEM due to charging effects)
        if light_intensity > 0.0 && self.config.lighting.enable_brightness {
            let brightness_range = self.config.lighting.brightness_range;
            let adjusted_range = (
                brightness_range.0 * light_intensity as f32,
                brightness_range.1 * light_intensity as f32,
            );
            
            composite = composite.add_augmentation(Box::new(LocalIlluminationAugmentation::new(
                (adjusted_range.0 as f64, adjusted_range.1 as f64),
                GradientType::Radial,
            )));

            let brightness_intensity = adjusted_range.0 / 2.0;
            composite = composite.add_augmentation(Box::new(BrightnessAugmentation::new((
                brightness_intensity as f64,
                -brightness_intensity as f64,
            ))));
        }

        // Add noise (shot noise is very common in SEM)
        if noise_intensity > 0.0 {
            if self.config.noise.enable_gaussian {
                let gaussian_range = self.config.noise.gaussian_std_range;
                let adjusted_range = (
                    gaussian_range.0 * noise_intensity as f32 * 0.1,
                    gaussian_range.1 * noise_intensity as f32 * 0.1,
                );
                composite = composite.add_augmentation(Box::new(ShotNoiseAugmentation::new((
                    adjusted_range.0 as f64,
                    adjusted_range.1 as f64,
                ))));
            }

            // Occasional salt-and-pepper from detector issues
            if self.config.noise.enable_salt_pepper && self.base.clone().random_bool(0.3) {
                let salt_pepper_ratio = self.config.noise.salt_pepper_ratio * noise_intensity as f32;
                composite = composite.add_augmentation(Box::new(SaltPepperNoiseAugmentation::new(
                    ((salt_pepper_ratio * 0.1) as f64, (salt_pepper_ratio * 0.5) as f64)
                )));
            }
        }

        // Add slight blur (due to beam size and astigmatism)
        if blur_intensity > 0.0 && self.config.blur.enable_gaussian {
            let sigma_range = self.config.blur.gaussian_sigma_range;
            let adjusted_range = (
                sigma_range.0 * blur_intensity as f32 * 0.2,
                sigma_range.1 * blur_intensity as f32 * 0.2,
            );
            composite = composite.add_augmentation(Box::new(GaussianBlurAugmentation::new((
                adjusted_range.0 as f64,
                adjusted_range.1 as f64,
            ))));
        }

        // Add slight distortion (from magnetic field variations)
        if distortion_intensity > 0.0 && self.config.distortion.enable_barrel && self.base.clone().random_bool(0.2) {
            let barrel_range = self.config.distortion.barrel_distortion_range;
            let adjusted_range = (
                barrel_range.0 * distortion_intensity as f32,
                barrel_range.1 * distortion_intensity as f32,
            );
            composite = composite.add_augmentation(Box::new(BarrelDistortionAugmentation::new((
                adjusted_range.0 as f64,
                adjusted_range.1 as f64,
            ))));
        }

        // Apply the composite augmentation
        let mut result = composite.apply(image)?;

        // Update augmentation info
        result.augmentations_applied = vec!["realistic_sem".to_string()];
        result
            .augmentation_params
            .insert("intensity".to_string(), json!(self.intensity));

        Ok(result)
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Complex realistic augmentation is not invertible
    }

    fn description(&self) -> String {
        format!(
            "Realistic SEM imaging effects with intensity {:.2}",
            self.intensity
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("realistic_sem"));
        params.insert("intensity".to_string(), json!(self.intensity));
        params
    }
}

/// Drift simulation augmentation (simulates stage drift during long acquisitions)
pub struct DriftSimulationAugmentation {
    base: AugmentationBase,
    max_drift_pixels: f64,
    drift_type: DriftType,
}

#[derive(Clone, Debug)]
pub enum DriftType {
    Linear,  // Constant drift direction
    Random,  // Random walk
    Thermal, // Exponential settling
}

impl DriftSimulationAugmentation {
    pub fn new(max_drift_pixels: f64, drift_type: DriftType) -> Self {
        Self {
            base: AugmentationBase::default(),
            max_drift_pixels,
            drift_type,
        }
    }
}

impl ImageAugmentation for DriftSimulationAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();

        let (dx, dy) = match self.drift_type {
            DriftType::Linear => {
                let angle = base.random_in_range(0.0, 2.0 * std::f64::consts::PI);
                let magnitude = base.random_in_range(0.0, self.max_drift_pixels);
                (magnitude * angle.cos(), magnitude * angle.sin())
            }
            DriftType::Random => (
                base.random_in_range(-self.max_drift_pixels, self.max_drift_pixels),
                base.random_in_range(-self.max_drift_pixels, self.max_drift_pixels),
            ),
            DriftType::Thermal => {
                // Exponential settling pattern
                let settle_factor = base.random_in_range(0.1, 0.9);
                let final_dx = base.random_in_range(-self.max_drift_pixels, self.max_drift_pixels);
                let final_dy = base.random_in_range(-self.max_drift_pixels, self.max_drift_pixels);
                (final_dx * settle_factor, final_dy * settle_factor)
            }
        };

        // Apply translation using existing transformer
        use crate::data::ImageTransformer;
        let drifted_image =
            ImageTransformer::translate(&image_to_grayimage(image)?, dx as i32, dy as i32)?;
        let output = grayimage_to_mat(&drifted_image)?;

        let ground_truth = crate::pipeline::GroundTruth {
            expected_location: crate::pipeline::SerializableRect {
                x: -dx as i32,
                y: -dy as i32,
                width: image.cols(),
                height: image.rows(),
            },
            transformation: crate::pipeline::TransformParams {
                translation: (-dx as f32, -dy as f32),
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
            augmentations_applied: vec!["drift_simulation".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("drift_x".to_string(), json!(dx));
                params.insert("drift_y".to_string(), json!(dy));
                params.insert(
                    "drift_type".to_string(),
                    json!(format!("{:?}", self.drift_type)),
                );
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        Some(identity_transform()) // Translation is invertible
    }

    fn description(&self) -> String {
        format!(
            "Stage drift simulation ({:?}) up to {:.1} pixels",
            self.drift_type, self.max_drift_pixels
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("drift_simulation"));
        params.insert("max_drift_pixels".to_string(), json!(self.max_drift_pixels));
        params.insert(
            "drift_type".to_string(),
            json!(format!("{:?}", self.drift_type)),
        );
        params
    }
}
