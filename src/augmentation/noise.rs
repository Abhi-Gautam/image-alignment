use crate::augmentation::base::*;
use crate::config::NoiseConfig;
use crate::pipeline::{AugmentedImage, GroundTruth, ImageAugmentation, Transform};
use crate::Result;
use opencv::core::{Mat, Scalar, RNG};
use opencv::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Gaussian noise augmentation
pub struct GaussianNoiseAugmentation {
    base: AugmentationBase,
    mean: f64,
    config: NoiseConfig,
}

impl GaussianNoiseAugmentation {
    pub fn new(sigma_range: (f64, f64)) -> Self {
        let config = NoiseConfig {
            gaussian_std_range: (sigma_range.0 as f32, sigma_range.1 as f32),
            ..Default::default()
        };
        Self {
            base: AugmentationBase::default(),
            mean: 0.0,
            config,
        }
    }

    pub fn from_config(config: NoiseConfig) -> Self {
        Self {
            base: AugmentationBase::default(),
            mean: 0.0,
            config,
        }
    }

    pub fn with_mean(mut self, mean: f64) -> Self {
        self.mean = mean;
        self
    }
}

impl ImageAugmentation for GaussianNoiseAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let sigma = base.random_in_range(self.config.gaussian_std_range.0 as f64, self.config.gaussian_std_range.1 as f64);

        // Create noise matrix
        let mut noise = Mat::zeros(image.rows(), image.cols(), image.typ())?.to_mat()?;
        let mut rng = RNG::new(base.rng.gen())?;
        rng.fill(
            &mut noise,
            opencv::core::RNG_NORMAL,
            &opencv::core::Scalar::all(self.mean),
            &opencv::core::Scalar::all(sigma),
            true,
        )?;

        // Add noise to image
        let mut output = Mat::default();
        opencv::core::add(image, &noise, &mut output, &opencv::core::no_array(), -1)?;

        // Clamp to valid range
        let mut clamped = Mat::default();
        opencv::core::min(&output, &Scalar::all(255.0), &mut clamped)?;
        opencv::core::max(&clamped, &Scalar::all(0.0), &mut output)?;
        let output = output.to_u8()?;

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
            augmentations_applied: vec!["gaussian_noise".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("mean".to_string(), json!(self.mean));
                params.insert("sigma".to_string(), json!(sigma));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Noise is not invertible
    }

    fn description(&self) -> String {
        format!("Gaussian noise with sigma in range {:?}", self.config.gaussian_std_range)
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("gaussian_noise"));
        params.insert("mean".to_string(), json!(self.mean));
        params.insert("sigma_range".to_string(), json!(self.config.gaussian_std_range));
        params
    }
}

/// Salt and pepper noise augmentation
pub struct SaltPepperNoiseAugmentation {
    base: AugmentationBase,
    salt_vs_pepper: f64,           // Ratio of salt to pepper (0.5 = equal)
    config: NoiseConfig,
}

impl SaltPepperNoiseAugmentation {
    pub fn new(noise_ratio_range: (f64, f64)) -> Self {
        let config = NoiseConfig {
            salt_pepper_ratio: (noise_ratio_range.1 / 2.0) as f32,
            ..Default::default()
        };
        Self {
            base: AugmentationBase::default(),
            salt_vs_pepper: 0.5,
            config,
        }
    }

    pub fn from_config(config: NoiseConfig) -> Self {
        Self {
            base: AugmentationBase::default(),
            salt_vs_pepper: 0.5,
            config,
        }
    }

    pub fn with_salt_vs_pepper(mut self, ratio: f64) -> Self {
        self.salt_vs_pepper = ratio.clamp(0.0, 1.0);
        self
    }
}

impl ImageAugmentation for SaltPepperNoiseAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let noise_ratio = base.random_in_range(0.001, self.config.salt_pepper_ratio as f64 * 2.0);

        let mut output = image.clone();
        let total_pixels = (image.rows() * image.cols()) as usize;
        let num_noise_pixels = (total_pixels as f64 * noise_ratio) as usize;

        // Generate random pixel locations
        for _ in 0..num_noise_pixels {
            let x = base.rng.gen_range(0..image.cols());
            let y = base.rng.gen_range(0..image.rows());

            // Decide salt or pepper
            let value = if base.random_bool(self.salt_vs_pepper) {
                255u8 // Salt
            } else {
                0u8 // Pepper
            };

            *output.at_2d_mut::<u8>(y, x)? = value;
        }

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
            augmentations_applied: vec!["salt_pepper_noise".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("noise_ratio".to_string(), json!(noise_ratio));
                params.insert("salt_vs_pepper".to_string(), json!(self.salt_vs_pepper));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Noise is not invertible
    }

    fn description(&self) -> String {
        format!(
            "Salt and pepper noise with ratio in range {:?}",
            (0.001, self.config.salt_pepper_ratio as f64 * 2.0)
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("salt_pepper_noise"));
        params.insert(
            "noise_ratio_range".to_string(),
            json!((0.001, self.config.salt_pepper_ratio as f64 * 2.0)),
        );
        params.insert("salt_vs_pepper".to_string(), json!(self.salt_vs_pepper));
        params
    }
}

/// Speckle noise augmentation (multiplicative noise common in imaging systems)
pub struct SpeckleNoiseAugmentation {
    base: AugmentationBase,
    variance_range: (f64, f64),
}

impl SpeckleNoiseAugmentation {
    pub fn new(variance_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            variance_range,
        }
    }
}

impl ImageAugmentation for SpeckleNoiseAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let variance = base.random_in_range(self.variance_range.0, self.variance_range.1);

        // Speckle noise: output = image + image * noise
        let image_f32 = image.to_f32()?;

        // Generate multiplicative noise
        let mut noise = Mat::zeros(image.rows(), image.cols(), opencv::core::CV_32F)?.to_mat()?;
        let normal = Normal::new(0.0, variance.sqrt()).unwrap();

        for y in 0..image.rows() {
            for x in 0..image.cols() {
                let noise_val = normal.sample(&mut base.rng) as f32;
                *noise.at_2d_mut::<f32>(y, x)? = noise_val;
            }
        }

        // Apply multiplicative noise
        let mut multiplied = Mat::default();
        opencv::core::multiply(&image_f32, &noise, &mut multiplied, 1.0, -1)?;

        let mut output = Mat::default();
        opencv::core::add(
            &image_f32,
            &multiplied,
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
            augmentations_applied: vec!["speckle_noise".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("variance".to_string(), json!(variance));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Noise is not invertible
    }

    fn description(&self) -> String {
        format!(
            "Speckle noise with variance in range {:?}",
            self.variance_range
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("speckle_noise"));
        params.insert("variance_range".to_string(), json!(self.variance_range));
        params
    }
}

/// Shot noise augmentation (Poisson noise - common in low-light imaging)
pub struct ShotNoiseAugmentation {
    base: AugmentationBase,
    scale_range: (f64, f64), // Controls intensity of shot noise
}

impl ShotNoiseAugmentation {
    pub fn new(scale_range: (f64, f64)) -> Self {
        Self {
            base: AugmentationBase::default(),
            scale_range,
        }
    }
}

impl ImageAugmentation for ShotNoiseAugmentation {
    fn apply(&self, image: &Mat) -> Result<AugmentedImage> {
        let mut base = self.base.clone();
        let scale = base.random_in_range(self.scale_range.0, self.scale_range.1);

        let mut output = Mat::zeros(image.rows(), image.cols(), image.typ())?.to_mat()?;

        // Apply Poisson noise to each pixel
        for y in 0..image.rows() {
            for x in 0..image.cols() {
                let pixel_val = *image.at_2d::<u8>(y, x)? as f64;

                // Scale pixel value for Poisson parameter
                let lambda = pixel_val * scale;

                // Approximate Poisson with Gaussian for large lambda
                let noise_val = if lambda > 10.0 {
                    let normal = Normal::new(lambda, lambda.sqrt()).unwrap();
                    normal.sample(&mut base.rng) / scale
                } else {
                    // For small lambda, use actual Poisson sampling
                    let mut poisson_val = 0.0;
                    let l = (-lambda).exp();
                    let mut p = 1.0;

                    while p > l {
                        poisson_val += 1.0;
                        p *= base.rng.gen::<f64>();
                    }

                    (poisson_val - 1.0) / scale
                };

                *output.at_2d_mut::<u8>(y, x)? = noise_val.clamp(0.0, 255.0) as u8;
            }
        }

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
            augmentations_applied: vec!["shot_noise".to_string()],
            augmentation_params: {
                let mut params = HashMap::new();
                params.insert("scale".to_string(), json!(scale));
                params
            },
        })
    }

    fn get_inverse_transform(&self) -> Option<Transform> {
        None // Noise is not invertible
    }

    fn description(&self) -> String {
        format!(
            "Shot (Poisson) noise with scale in range {:?}",
            self.scale_range
        )
    }

    fn get_params(&self) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        params.insert("type".to_string(), json!("shot_noise"));
        params.insert("scale_range".to_string(), json!(self.scale_range));
        params
    }
}
