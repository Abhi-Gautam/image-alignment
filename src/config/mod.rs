use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub mod manipulation;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Config {
    pub algorithms: AlgorithmConfig,
    pub image: ImageConfig,
    pub augmentation: AugmentationConfig,
    pub testing: TestingConfig,
    pub dashboard: DashboardConfig,
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct AlgorithmConfig {
    pub orb: OrbConfig,
    pub template: TemplateConfig,
    pub akaze: AkazeConfig,
    pub sift: SiftConfig,
    pub ransac: RansacConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbConfig {
    pub max_features: i32,
    pub scale_factor: f32,
    pub n_levels: i32,
    pub edge_threshold: i32,
    pub first_level: i32,
    pub wta_k: i32,
    pub score_type: i32,
    pub patch_size: i32,
    pub fast_threshold: i32,
    pub distance_threshold: f32,
    pub ratio_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    pub distance_threshold: f32,
    pub match_threshold: f32,
    pub normalization_factor: f32,
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkazeConfig {
    pub threshold: f64,
    pub octaves: i32,
    pub octave_layers: i32,
    pub diffusivity: i32,
    pub max_features: i32,
    pub distance_threshold: f32,
    pub ratio_threshold: f32,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiftConfig {
    pub n_features: i32,
    pub n_octave_layers: i32,
    pub contrast_threshold: f64,
    pub edge_threshold: f64,
    pub sigma: f64,
    pub distance_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RansacConfig {
    pub max_iterations: i32,
    pub inlier_threshold: f32,
    pub min_inliers: i32,
    pub success_probability: f32,
    pub outlier_ratio: f32,
    pub early_exit_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    pub min_size: u32,
    pub max_size: u32,
    pub default_patch_sizes: Vec<u32>,
    pub variance_threshold: f32,
    pub min_features_for_alignment: u32,
    pub pyramid_levels: i32,
    pub edge_detection_threshold: f32,
    pub gaussian_blur_sigma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct AugmentationConfig {
    pub noise: NoiseConfig,
    pub blur: BlurConfig,
    pub lighting: LightingConfig,
    pub geometric: GeometricConfig,
    pub distortion: DistortionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    pub gaussian_std_range: (f32, f32),
    pub salt_pepper_ratio: f32,
    pub uniform_range: (f32, f32),
    pub enable_gaussian: bool,
    pub enable_salt_pepper: bool,
    pub enable_uniform: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlurConfig {
    pub gaussian_kernel_range: (i32, i32),
    pub gaussian_sigma_range: (f32, f32),
    pub motion_length_range: (i32, i32),
    pub motion_angle_range: (f32, f32),
    pub enable_gaussian: bool,
    pub enable_motion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingConfig {
    pub brightness_range: (f32, f32),
    pub contrast_range: (f32, f32),
    pub gamma_range: (f32, f32),
    pub exposure_range: (f32, f32),
    pub enable_brightness: bool,
    pub enable_contrast: bool,
    pub enable_gamma: bool,
    pub enable_exposure: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricConfig {
    pub rotation_range: (f32, f32),
    pub scale_range: (f32, f32),
    pub translation_range: (f32, f32),
    pub shear_range: (f32, f32),
    pub enable_rotation: bool,
    pub enable_scale: bool,
    pub enable_translation: bool,
    pub enable_shear: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistortionConfig {
    pub barrel_distortion_range: (f32, f32),
    pub pincushion_distortion_range: (f32, f32),
    pub perspective_distortion_strength: f32,
    pub enable_barrel: bool,
    pub enable_pincushion: bool,
    pub enable_perspective: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingConfig {
    pub accuracy_threshold_pixels: f32,
    pub rotation_accuracy_threshold_degrees: f32,
    pub scale_accuracy_threshold: f32,
    pub min_confidence_threshold: f32,
    pub max_processing_time_ms: f32,
    pub test_repetitions: u32,
    pub validation_sample_size: u32,
    pub benchmark_iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub default_port: u16,
    pub request_timeout_ms: u64,
    pub max_concurrent_requests: usize,
    pub static_file_cache_duration_secs: u64,
    pub enable_cors: bool,
    pub max_upload_size_mb: usize,
    pub results_per_page: usize,
    pub chart_colors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub success_criteria: SuccessCriteria,
    pub confidence_thresholds: HashMap<String, f32>,
    pub performance_targets: PerformanceTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub translation_accuracy_px: f32,
    pub min_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub target_processing_time_ms: f32,
    pub target_accuracy_px: f32,
    pub target_success_rate: f32,
}



impl Default for OrbConfig {
    fn default() -> Self {
        Self {
            max_features: 500,
            scale_factor: 1.5,
            n_levels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            score_type: 0,
            patch_size: 31,
            fast_threshold: 20,
            distance_threshold: 100.0,
            ratio_threshold: 0.7,
        }
    }
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 0.8,
            match_threshold: 0.7,
            normalization_factor: 1.0,
            confidence_threshold: 0.5,
        }
    }
}

impl Default for AkazeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.001,
            octaves: 4,
            octave_layers: 4,
            diffusivity: 1,
            max_features: 1000,
            distance_threshold: 100.0,
            ratio_threshold: 0.7,
        }
    }
}


impl Default for SiftConfig {
    fn default() -> Self {
        Self {
            n_features: 1000,
            n_octave_layers: 3,
            contrast_threshold: 0.02,
            edge_threshold: 5.0,
            sigma: 1.6,
            distance_threshold: 100.0,
        }
    }
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            inlier_threshold: 1.5,
            min_inliers: 2,
            success_probability: 0.99,
            outlier_ratio: 0.5,
            early_exit_threshold: 0.95,
        }
    }
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            min_size: 8,
            max_size: 10000,
            default_patch_sizes: vec![32, 64, 128],
            variance_threshold: 100.0,
            min_features_for_alignment: 10,
            pyramid_levels: 3,
            edge_detection_threshold: 50.0,
            gaussian_blur_sigma: 1.0,
        }
    }
}


impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            gaussian_std_range: (0.0, 10.0),
            salt_pepper_ratio: 0.05,
            uniform_range: (-10.0, 10.0),
            enable_gaussian: true,
            enable_salt_pepper: true,
            enable_uniform: true,
        }
    }
}

impl Default for BlurConfig {
    fn default() -> Self {
        Self {
            gaussian_kernel_range: (3, 15),
            gaussian_sigma_range: (0.5, 3.0),
            motion_length_range: (5, 20),
            motion_angle_range: (0.0, 360.0),
            enable_gaussian: true,
            enable_motion: true,
        }
    }
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            brightness_range: (-30.0, 30.0),
            contrast_range: (0.7, 1.3),
            gamma_range: (0.7, 1.3),
            exposure_range: (-1.0, 1.0),
            enable_brightness: true,
            enable_contrast: true,
            enable_gamma: true,
            enable_exposure: true,
        }
    }
}

impl Default for GeometricConfig {
    fn default() -> Self {
        Self {
            rotation_range: (-30.0, 30.0),
            scale_range: (0.8, 1.2),
            translation_range: (-50.0, 50.0),
            shear_range: (-0.2, 0.2),
            enable_rotation: true,
            enable_scale: true,
            enable_translation: true,
            enable_shear: true,
        }
    }
}

impl Default for DistortionConfig {
    fn default() -> Self {
        Self {
            barrel_distortion_range: (-0.3, -0.1),
            pincushion_distortion_range: (0.1, 0.3),
            perspective_distortion_strength: 0.1,
            enable_barrel: true,
            enable_pincushion: true,
            enable_perspective: true,
        }
    }
}

impl Default for TestingConfig {
    fn default() -> Self {
        Self {
            accuracy_threshold_pixels: 0.1,
            rotation_accuracy_threshold_degrees: 0.5,
            scale_accuracy_threshold: 0.01,
            min_confidence_threshold: 0.5,
            max_processing_time_ms: 50.0,
            test_repetitions: 3,
            validation_sample_size: 100,
            benchmark_iterations: 10,
        }
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            default_port: 3000,
            request_timeout_ms: 5000,
            max_concurrent_requests: 100,
            static_file_cache_duration_secs: 3600,
            enable_cors: true,
            max_upload_size_mb: 10,
            results_per_page: 20,
            chart_colors: vec![
                "#3498db".to_string(),
                "#e74c3c".to_string(),
                "#2ecc71".to_string(),
                "#f39c12".to_string(),
                "#9b59b6".to_string(),
                "#1abc9c".to_string(),
            ],
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        let mut confidence_thresholds = HashMap::new();
        confidence_thresholds.insert("orb".to_string(), 0.5);
        confidence_thresholds.insert("template".to_string(), 0.7);
        confidence_thresholds.insert("akaze".to_string(), 0.6);
        confidence_thresholds.insert("sift".to_string(), 0.6);

        Self {
            success_criteria: SuccessCriteria::default(),
            confidence_thresholds,
            performance_targets: PerformanceTargets::default(),
        }
    }
}

impl Default for SuccessCriteria {
    fn default() -> Self {
        Self {
            translation_accuracy_px: 12.8, // 20% of 64px patch
            min_confidence: 0.3,
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_processing_time_ms: 25.0,
            target_accuracy_px: 0.05,
            target_success_rate: 0.95,
        }
    }
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        
        if content.trim_start().starts_with('{') {
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(toml::from_str(&content)?)
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P, format: ConfigFormat) -> Result<(), Box<dyn std::error::Error>> {
        let content = match format {
            ConfigFormat::Json => serde_json::to_string_pretty(self)?,
            ConfigFormat::Toml => toml::to_string_pretty(self)?,
        };
        
        fs::write(path, content)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.algorithms.orb.max_features <= 0 {
            errors.push("ORB max_features must be positive".to_string());
        }

        if self.algorithms.orb.scale_factor <= 1.0 {
            errors.push("ORB scale_factor must be greater than 1.0".to_string());
        }

        if self.image.min_size >= self.image.max_size {
            errors.push("Image min_size must be less than max_size".to_string());
        }

        if self.testing.accuracy_threshold_pixels < 0.0 {
            errors.push("Accuracy threshold must be non-negative".to_string());
        }

        if self.dashboard.default_port == 0 {
            errors.push("Dashboard port must be valid".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConfigFormat {
    Json,
    Toml,
}

pub fn load_config_or_default(config_path: Option<&str>) -> Config {
    match config_path {
        Some(path) => {
            match Config::load_from_file(path) {
                Ok(config) => {
                    if let Err(errors) = config.validate() {
                        eprintln!("Configuration validation errors:");
                        for error in errors {
                            eprintln!("  - {}", error);
                        }
                        eprintln!("Using default configuration instead.");
                        Config::default()
                    } else {
                        config
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load config from '{}': {}", path, e);
                    eprintln!("Using default configuration.");
                    Config::default()
                }
            }
        }
        None => Config::default(),
    }
}