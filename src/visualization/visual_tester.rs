use crate::algorithms::*;
use crate::config::Config;
use crate::logging::{TestSessionSpan, new_correlation_id, set_correlation_id};
use crate::pipeline::{AlignmentAlgorithm, AlignmentResult};
use crate::utils::image_conversion::grayimage_to_mat;
use image::{imageops, GrayImage, Luma, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{info, debug, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub test_id: String,
    pub session_id: String,
    pub correlation_id: String,
    pub algorithm_name: String,
    pub original_image_path: String,
    pub patch_info: PatchInfo,
    pub transformation_applied: TransformationInfo,
    pub alignment_result: AlignmentResult,
    pub visual_outputs: VisualOutputs,
    pub performance_metrics: PerformanceMetrics,
    pub log_file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchInfo {
    pub size: (u32, u32),
    pub location: (u32, u32),
    pub patch_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationInfo {
    pub noise_type: String,
    pub noise_parameters: String,
    pub rotation_deg: f32,
    pub translation: (i32, i32),
    pub scale_factor: f32,
    pub transformed_patch_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualOutputs {
    pub overlay_result_path: String,
    pub side_by_side_path: String,
    pub error_heatmap_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub translation_error_px: f32,
    pub processing_time_ms: f32,
    pub confidence_score: f32,
    pub success: bool,
}

pub struct VisualTester {
    pub output_dir: PathBuf,
    pub algorithms: Vec<Box<dyn AlignmentAlgorithm>>,
    pub config: Config,
}

impl VisualTester {
    pub fn new(output_dir: PathBuf) -> Self {
        // Create output directory
        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

        // Initialize all algorithms
        let mut algorithms: Vec<Box<dyn AlignmentAlgorithm>> = Vec::new();
        algorithms.push(Box::new(OpenCVTemplateMatcher::new_ncc()));
        algorithms.push(Box::new(OpenCVTemplateMatcher::new_ssd()));
        if let Ok(orb) = OpenCVORB::new() {
            algorithms.push(Box::new(orb));
        }

        // Add new algorithms if they can be created successfully
        if let Ok(akaze) = OpenCVAKAZE::new() {
            algorithms.push(Box::new(akaze));
        }


        if let Ok(sift) = OpenCVSIFT::new() {
            algorithms.push(Box::new(sift));
        }

        Self {
            output_dir,
            algorithms,
            config: Config::default(),
        }
    }

    pub fn with_config(output_dir: PathBuf, config: Config) -> Self {
        // Create output directory
        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

        // Initialize all algorithms with config
        let mut algorithms: Vec<Box<dyn AlignmentAlgorithm>> = Vec::new();
        algorithms.push(Box::new(OpenCVTemplateMatcher::with_config(
            crate::algorithms::opencv_template::TemplateMatchMode::NormalizedCrossCorrelation, 
            config.algorithms.template.clone()
        )));
        algorithms.push(Box::new(OpenCVTemplateMatcher::with_config(
            crate::algorithms::opencv_template::TemplateMatchMode::SumOfSquaredDifferences,
            config.algorithms.template.clone()
        )));
        if let Ok(orb) = OpenCVORB::with_config(config.algorithms.orb.clone()) {
            algorithms.push(Box::new(orb));
        }

        // Add new algorithms if they can be created successfully
        if let Ok(akaze) = OpenCVAKAZE::with_config(config.algorithms.akaze.clone()) {
            algorithms.push(Box::new(akaze));
        }


        if let Ok(sift) = OpenCVSIFT::with_config(config.algorithms.sift.clone()) {
            algorithms.push(Box::new(sift));
        }

        Self {
            output_dir,
            algorithms,
            config,
        }
    }

    /// Run comprehensive visual tests on a SEM image
    pub fn run_comprehensive_test(
        &mut self,
        sem_image_path: &Path,
        patch_sizes: Option<&[u32]>,
        scenarios: Option<&[String]>,
    ) -> crate::Result<Vec<TestReport>> {
        // Create test session with correlation tracking
        let session_id = Uuid::new_v4();
        let session_correlation_id = new_correlation_id();
        
        let test_session_span = TestSessionSpan::new("visual_test", session_id);
        let _session_guard = test_session_span.enter();
        
        // Create session-specific log directory
        let session_log_dir = self.output_dir.join("logs").join(format!("session_{}", session_id));
        std::fs::create_dir_all(&session_log_dir)?;
        
        info!(
            session_id = %session_id,
            correlation_id = %session_correlation_id,
            sem_image_path = %sem_image_path.display(),
            output_dir = %self.output_dir.display(),
            "Starting comprehensive visual test session"
        );

        // Load the SEM image
        let sem_image = image::open(sem_image_path)?.to_luma8();
        
        info!(
            image_width = sem_image.width(),
            image_height = sem_image.height(),
            "Loaded SEM image successfully"
        );
        
        // Record test configuration in span
        let config_json = serde_json::json!({
            "patch_sizes": patch_sizes.unwrap_or(&[32, 64, 128]),
            "scenarios": scenarios.unwrap_or(&["default".to_string()]),
            "algorithms": self.algorithms.iter().map(|a| a.name()).collect::<Vec<_>>(),
            "output_directory": self.output_dir.to_string_lossy()
        });
        test_session_span.record_config("comprehensive_visual_test", config_json);

        // Extract multiple patches of different sizes
        let default_patch_sizes = vec![32, 64, 128];
        let patch_sizes = patch_sizes.unwrap_or(&default_patch_sizes);
        let mut all_reports = Vec::new();

        let mut total_tests = 0;
        let mut successful_tests = 0;

        for &patch_size in patch_sizes {
            debug!(
                patch_size = patch_size,
                "Starting tests for patch size"
            );

            // Extract 3 patches of this size
            let patches = self.extract_good_patches(&sem_image, patch_size, 3)?;

            if patches.is_empty() {
                warn!(
                    patch_size = patch_size,
                    image_width = sem_image.width(),
                    image_height = sem_image.height(),
                    "Skipping patch size - image too small"
                );
                continue;
            }

            for (patch_idx, (patch, x, y)) in patches.into_iter().enumerate() {
                let test_base_id = format!("{}x{}_patch{}", patch_size, patch_size, patch_idx + 1);

                debug!(
                    test_base_id = %test_base_id,
                    patch_location = format!("({}, {})", x, y),
                    "Extracted patch for testing"
                );

                // Test different noise and transformation scenarios
                let test_scenarios = self.create_test_scenarios(scenarios);

                for scenario in test_scenarios {
                    // Apply transformations and noise to the patch
                    let transformed_patch =
                        self.apply_transformation_and_noise(&patch, &scenario)?;

                    // Test all algorithms
                    for algorithm in &self.algorithms {
                        // Create unique correlation ID for each test iteration
                        let test_correlation_id = new_correlation_id();
                        set_correlation_id(test_correlation_id);
                        
                        let test_id =
                            format!("{}_{}_{}", test_base_id, scenario.name, algorithm.name());

                        debug!(
                            test_id = %test_id,
                            algorithm = algorithm.name(),
                            scenario = %scenario.name,
                            correlation_id = %test_correlation_id,
                            "Starting individual algorithm test"
                        );

                        // Record test iteration in session span
                        test_session_span.record_iteration(
                            total_tests,
                            algorithm.name(),
                            true // We'll update this based on actual result
                        );

                        let report = self.run_single_test(
                            &test_id,
                            &session_id.to_string(),
                            &test_correlation_id.to_string(),
                            sem_image_path,
                            &sem_image,
                            &patch,
                            (x, y),
                            &transformed_patch,
                            &scenario,
                            algorithm.as_ref(),
                        )?;

                        // Update test counters
                        total_tests += 1;
                        if report.performance_metrics.success {
                            successful_tests += 1;
                        }

                        debug!(
                            test_id = %test_id,
                            success = report.performance_metrics.success,
                            confidence = report.alignment_result.confidence,
                            "Test iteration completed"
                        );

                        all_reports.push(report);
                    }
                }
            }
        }

        // Record session completion in span
        test_session_span.record_completion(total_tests, successful_tests);

        // Generate summary report with session correlation
        info!(
            session_id = %session_id,
            total_tests = total_tests,
            successful_tests = successful_tests,
            success_rate = if total_tests > 0 { (successful_tests as f64) / (total_tests as f64) * 100.0 } else { 0.0 },
            "Visual test session completed"
        );

        self.generate_summary_report(&all_reports, &session_id.to_string())?;

        // Save session metadata
        let session_metadata = serde_json::json!({
            "session_id": session_id.to_string(),
            "correlation_id": session_correlation_id.to_string(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": if total_tests > 0 { (successful_tests as f64) / (total_tests as f64) * 100.0 } else { 0.0 },
            "algorithms_tested": self.algorithms.iter().map(|a| a.name()).collect::<Vec<_>>(),
            "patch_sizes": patch_sizes,
            "timestamp": chrono::Utc::now().to_rfc3339()
        });

        let session_file = self.output_dir.join(format!("session_{}_metadata.json", session_id));
        std::fs::write(session_file, serde_json::to_string_pretty(&session_metadata)?)?;

        Ok(all_reports)
    }


    /// Extract good patches with sufficient texture
    fn extract_good_patches(
        &self,
        image: &GrayImage,
        patch_size: u32,
        count: u32,
    ) -> crate::Result<Vec<(GrayImage, u32, u32)>> {
        let mut patches = Vec::new();
        let margin = patch_size / 2;
        let min_variance = 100.0;

        for attempt in 0..100 {
            // Ensure we don't divide by zero
            let x_range = if image.width() > patch_size + margin {
                image.width() - patch_size - margin
            } else {
                1 // fallback to avoid division by zero
            };
            let y_range = if image.height() > patch_size + margin {
                image.height() - patch_size - margin
            } else {
                1 // fallback to avoid division by zero
            };

            let x = margin + (attempt * 47) % x_range;
            let y = margin + (attempt * 37) % y_range;

            // Extract patch
            let patch = imageops::crop_imm(image, x, y, patch_size, patch_size).to_image();

            // Check variance (texture quality)
            let variance = self.calculate_patch_variance(&patch);
            if variance > min_variance {
                patches.push((patch, x, y));
                if patches.len() >= count as usize {
                    break;
                }
            }
        }

        if patches.is_empty() {
            return Err(anyhow::anyhow!(
                "No suitable patches found with sufficient texture"
            ));
        }

        Ok(patches)
    }

    fn calculate_patch_variance(&self, patch: &GrayImage) -> f32 {
        let mean: f32 = patch.pixels().map(|p| p[0] as f32).sum::<f32>()
            / (patch.width() * patch.height()) as f32;
        let variance: f32 = patch
            .pixels()
            .map(|p| {
                let diff = p[0] as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / (patch.width() * patch.height()) as f32;
        variance
    }

    /// Create test scenarios (noise, blur, brightness, etc.)
    fn create_test_scenarios(&self, scenarios: Option<&[String]>) -> Vec<TestScenario> {
        let all_scenarios = vec![
            TestScenario {
                name: "clean".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 0.0,
                translation: (0, 0),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "translation_5px".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 0.0,
                translation: (5, 5),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "translation_10px".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 0.0,
                translation: (10, -10),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "rotation_10deg".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 10.0,
                translation: (0, 0),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "rotation_30deg".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 30.0,
                translation: (0, 0),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "gaussian_noise".to_string(),
                noise_type: NoiseType::Gaussian { sigma: 5.0 },
                rotation_deg: 0.0,
                translation: (0, 0),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "salt_pepper".to_string(),
                noise_type: NoiseType::SaltPepper { density: 0.005 },
                rotation_deg: 0.0,
                translation: (0, 0),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "gaussian_blur".to_string(),
                noise_type: NoiseType::GaussianBlur { sigma: 1.5 },
                rotation_deg: 0.0,
                translation: (0, 0),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "brightness_change".to_string(),
                noise_type: NoiseType::Brightness { delta: 20 },
                rotation_deg: 0.0,
                translation: (0, 0),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "scale_120".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 0.0,
                translation: (0, 0),
                scale_factor: 1.2,
            },
        ];
        
        // Filter scenarios based on user selection
        match scenarios {
            Some(selected_scenarios) => {
                all_scenarios
                    .into_iter()
                    .filter(|s| selected_scenarios.iter().any(|name| name == &s.name))
                    .collect()
            }
            None => all_scenarios,
        }
    }

    /// Apply transformation and noise to a patch
    fn apply_transformation_and_noise(
        &self,
        patch: &GrayImage,
        scenario: &TestScenario,
    ) -> crate::Result<GrayImage> {
        let mut result = patch.clone();

        // Apply scaling
        if (scenario.scale_factor - 1.0).abs() > 0.01 {
            let new_width = (patch.width() as f32 * scenario.scale_factor) as u32;
            let new_height = (patch.height() as f32 * scenario.scale_factor) as u32;
            result = imageops::resize(
                &result,
                new_width,
                new_height,
                imageops::FilterType::Gaussian,
            );
        }

        // Apply rotation
        if scenario.rotation_deg.abs() > 0.1 {
            result = self.rotate_image(&result, scenario.rotation_deg)?;
        }

        // Apply noise
        result = self.apply_noise(&result, &scenario.noise_type)?;

        Ok(result)
    }

    fn rotate_image(&self, image: &GrayImage, angle_deg: f32) -> crate::Result<GrayImage> {
        let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let (width, height) = (image.width(), image.height());
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        let mut rotated = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                // Translate to center, rotate, translate back
                let x_centered = x as f32 - center_x;
                let y_centered = y as f32 - center_y;
                let x_rot = x_centered * cos_a + y_centered * sin_a + center_x;
                let y_rot = -x_centered * sin_a + y_centered * cos_a + center_y;

                // Bilinear interpolation
                let pixel_value = self.bilinear_sample(image, x_rot, y_rot);
                rotated.put_pixel(x, y, Luma([pixel_value]));
            }
        }

        Ok(rotated)
    }

    fn bilinear_sample(&self, image: &GrayImage, x: f32, y: f32) -> u8 {
        let (width, height) = (image.width(), image.height());

        if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
            return 128; // Gray background
        }

        let x1 = x.floor() as u32;
        let y1 = y.floor() as u32;
        let x2 = (x1 + 1).min(width - 1);
        let y2 = (y1 + 1).min(height - 1);

        let fx = x - x1 as f32;
        let fy = y - y1 as f32;

        let p11 = image.get_pixel(x1, y1)[0] as f32;
        let p12 = image.get_pixel(x1, y2)[0] as f32;
        let p21 = image.get_pixel(x2, y1)[0] as f32;
        let p22 = image.get_pixel(x2, y2)[0] as f32;

        let interpolated = p11 * (1.0 - fx) * (1.0 - fy)
            + p21 * fx * (1.0 - fy)
            + p12 * (1.0 - fx) * fy
            + p22 * fx * fy;

        interpolated.round() as u8
    }

    fn apply_noise(&self, image: &GrayImage, noise_type: &NoiseType) -> crate::Result<GrayImage> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut result = image.clone();

        match noise_type {
            NoiseType::None => {}
            NoiseType::Gaussian { sigma } => {
                for pixel in result.pixels_mut() {
                    let noise: f32 = rng.gen::<f32>() * 2.0 - 1.0; // -1 to 1
                    let noisy_val = pixel[0] as f32 + noise * sigma;
                    pixel[0] = noisy_val.clamp(0.0, 255.0) as u8;
                }
            }
            NoiseType::SaltPepper { density } => {
                for pixel in result.pixels_mut() {
                    if rng.gen::<f32>() < *density {
                        pixel[0] = if rng.gen::<bool>() { 255 } else { 0 };
                    }
                }
            }
            NoiseType::GaussianBlur { sigma } => {
                // Simple box blur approximation
                result = self.apply_box_blur(&result, (*sigma * 2.0) as u32)?;
            }
            NoiseType::Brightness { delta } => {
                for pixel in result.pixels_mut() {
                    let new_val = pixel[0] as i32 + delta;
                    pixel[0] = new_val.clamp(0, 255) as u8;
                }
            }
        }

        Ok(result)
    }

    fn apply_box_blur(&self, image: &GrayImage, radius: u32) -> crate::Result<GrayImage> {
        if radius == 0 {
            return Ok(image.clone());
        }

        let (width, height) = (image.width(), image.height());
        let mut result = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;
                let mut count = 0u32;

                for dy in -(radius as i32)..=(radius as i32) {
                    for dx in -(radius as i32)..=(radius as i32) {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                            sum += image.get_pixel(nx as u32, ny as u32)[0] as u32;
                            count += 1;
                        }
                    }
                }

                let average = if count > 0 { sum / count } else { 0 };
                result.put_pixel(x, y, Luma([average as u8]));
            }
        }

        Ok(result)
    }

    /// Run a single test and generate visual report
    #[allow(clippy::too_many_arguments)]
    fn run_single_test(
        &self,
        test_id: &str,
        session_id: &str,
        correlation_id: &str,
        sem_image_path: &Path,
        sem_image: &GrayImage,
        original_patch: &GrayImage,
        patch_location: (u32, u32),
        transformed_patch: &GrayImage,
        scenario: &TestScenario,
        algorithm: &dyn AlignmentAlgorithm,
    ) -> crate::Result<TestReport> {
        // Create test-specific directory and log files
        let test_dir = self.output_dir.join(test_id);
        std::fs::create_dir_all(&test_dir)?;
        
        // Create per-algorithm log file
        let algorithm_log_path = test_dir.join(format!("{}_algorithm.log", algorithm.name().to_lowercase()));
        let algorithm_log_path_str = algorithm_log_path.to_string_lossy().to_string();
        
        info!(
            test_id = %test_id,
            algorithm = algorithm.name(),
            patch_location = format!("({}, {})", patch_location.0, patch_location.1),
            scenario = %scenario.name,
            log_file = %algorithm_log_path_str,
            "Starting single algorithm test"
        );

        // Save original patch
        let patch_path = test_dir.join("1_original_patch.png");
        original_patch.save(&patch_path)?;
        
        debug!(
            patch_path = %patch_path.display(),
            patch_size = format!("{}x{}", original_patch.width(), original_patch.height()),
            "Saved original patch"
        );

        // Save transformed patch
        let transformed_patch_path = test_dir.join("2_transformed_patch.png");
        transformed_patch.save(&transformed_patch_path)?;
        
        debug!(
            transformed_patch_path = %transformed_patch_path.display(),
            transformation = %scenario.name,
            "Saved transformed patch"
        );

        // Convert patches to Mat format for the new pipeline
        let _original_mat = grayimage_to_mat(original_patch)?;
        let transformed_mat = grayimage_to_mat(transformed_patch)?;

        // Convert the full search image to Mat for alignment
        let search_mat = grayimage_to_mat(sem_image)?;

        // Set patch context for spatial logging
        let patch_context = crate::logging::PatchContext {
            extracted_from: patch_location,
            patch_size: (original_patch.width(), original_patch.height()),
            source_image_size: (sem_image.width(), sem_image.height()),
            variance: crate::data::PatchExtractor::calculate_variance(original_patch),
        };
        crate::logging::set_patch_context(patch_context);

        // Run alignment using the new signature
        // The algorithm searches for the transformed patch in the full SEM image
        // We expect it to find the patch near the original location since it's the same physical location
        debug!("Starting algorithm alignment");
        let mut alignment_result = algorithm.align(&search_mat, &transformed_mat)?;
        
        // Clear patch context after alignment
        crate::logging::clear_patch_context();
        
        info!(
            detected_location = format!("({}, {})", alignment_result.location.x, alignment_result.location.y),
            confidence = alignment_result.confidence,
            execution_time_ms = alignment_result.execution_time_ms,
            "Algorithm alignment completed"
        );

        // Update the transformation to include the expected location for proper error calculation
        if let Some(ref mut transform) = alignment_result.transformation {
            // The detected location minus the original patch location gives us the actual displacement
            let detected_displacement_x =
                alignment_result.location.x as f32 - patch_location.0 as f32;
            let detected_displacement_y =
                alignment_result.location.y as f32 - patch_location.1 as f32;
            transform.translation = (detected_displacement_x, detected_displacement_y);
        } else {
            // Create transformation info if not present
            alignment_result.transformation = Some(crate::pipeline::TransformParams {
                translation: (
                    alignment_result.location.x as f32 - patch_location.0 as f32,
                    alignment_result.location.y as f32 - patch_location.1 as f32,
                ),
                rotation_degrees: 0.0,
                scale: 1.0,
                skew: None,
            });
        }

        // Generate visualizations
        let visual_outputs = self.generate_visualizations(
            &test_dir,
            sem_image,
            original_patch,
            transformed_patch,
            patch_location,
            &alignment_result,
            scenario,
        )?;

        // Calculate performance metrics
        let patch_size = original_patch.width(); // Assuming square patches
        let performance_metrics = self.calculate_performance_metrics(&alignment_result, patch_location, patch_size, scenario);
        
        debug!(
            translation_error_px = performance_metrics.translation_error_px,
            success = performance_metrics.success,
            "Performance metrics calculated"
        );

        // Create test report with correlation tracking
        let report = TestReport {
            test_id: test_id.to_string(),
            session_id: session_id.to_string(),
            correlation_id: correlation_id.to_string(),
            algorithm_name: algorithm.name().to_string(),
            original_image_path: sem_image_path.to_string_lossy().to_string(),
            patch_info: PatchInfo {
                size: (original_patch.width(), original_patch.height()),
                location: patch_location,
                patch_path: patch_path.to_string_lossy().to_string(),
            },
            transformation_applied: TransformationInfo {
                noise_type: format!("{:?}", scenario.noise_type),
                noise_parameters: scenario.name.clone(),
                rotation_deg: scenario.rotation_deg,
                translation: scenario.translation,
                scale_factor: scenario.scale_factor,
                transformed_patch_path: transformed_patch_path.to_string_lossy().to_string(),
            },
            alignment_result,
            visual_outputs,
            performance_metrics,
            log_file_path: Some(algorithm_log_path_str),
        };

        // Save individual test report
        let report_path = test_dir.join("test_report.json");
        let report_json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&report_path, report_json)?;
        
        info!(
            test_id = %test_id,
            report_path = %report_path.display(),
            success = report.performance_metrics.success,
            confidence = report.alignment_result.confidence,
            "Test completed and report saved"
        );

        Ok(report)
    }

    /// Generate visual outputs for the test
    #[allow(clippy::too_many_arguments)]
    fn generate_visualizations(
        &self,
        test_dir: &Path,
        sem_image: &GrayImage,
        original_patch: &GrayImage,
        transformed_patch: &GrayImage,
        patch_location: (u32, u32),
        alignment_result: &AlignmentResult,
        _scenario: &TestScenario,
    ) -> crate::Result<VisualOutputs> {
        // Create side-by-side comparison
        let side_by_side_path = test_dir.join("3_side_by_side.png");
        self.create_side_by_side_image(original_patch, transformed_patch, &side_by_side_path)?;

        // Create overlay visualization showing alignment result
        let overlay_path = test_dir.join("4_alignment_overlay.png");
        self.create_alignment_overlay(
            sem_image,
            original_patch,
            transformed_patch,
            patch_location,
            alignment_result,
            &overlay_path,
        )?;

        // Create error heatmap
        let heatmap_path = test_dir.join("5_error_heatmap.png");
        self.create_error_heatmap(
            original_patch,
            transformed_patch,
            alignment_result,
            &heatmap_path,
        )?;

        Ok(VisualOutputs {
            side_by_side_path: side_by_side_path.to_string_lossy().to_string(),
            overlay_result_path: overlay_path.to_string_lossy().to_string(),
            error_heatmap_path: heatmap_path.to_string_lossy().to_string(),
        })
    }

    fn create_side_by_side_image(
        &self,
        img1: &GrayImage,
        img2: &GrayImage,
        output_path: &Path,
    ) -> crate::Result<()> {
        let total_width = img1.width() + img2.width() + 10; // 10px gap
        let max_height = img1.height().max(img2.height());

        let mut result = RgbImage::new(total_width, max_height);

        // Fill with white background
        for pixel in result.pixels_mut() {
            *pixel = Rgb([255, 255, 255]);
        }

        // Copy first image
        for y in 0..img1.height() {
            for x in 0..img1.width() {
                let gray_val = img1.get_pixel(x, y)[0];
                result.put_pixel(x, y, Rgb([gray_val, gray_val, gray_val]));
            }
        }

        // Copy second image
        let offset_x = img1.width() + 10;
        for y in 0..img2.height() {
            for x in 0..img2.width() {
                let gray_val = img2.get_pixel(x, y)[0];
                result.put_pixel(x + offset_x, y, Rgb([gray_val, gray_val, gray_val]));
            }
        }

        result.save(output_path)?;
        Ok(())
    }

    fn create_alignment_overlay(
        &self,
        sem_image: &GrayImage,
        original_patch: &GrayImage,
        _transformed_patch: &GrayImage,
        patch_location: (u32, u32),
        alignment_result: &AlignmentResult,
        output_path: &Path,
    ) -> crate::Result<()> {
        let mut overlay = RgbImage::new(sem_image.width(), sem_image.height());

        // Convert SEM image to RGB
        for y in 0..sem_image.height() {
            for x in 0..sem_image.width() {
                let gray_val = sem_image.get_pixel(x, y)[0];
                overlay.put_pixel(x, y, Rgb([gray_val, gray_val, gray_val]));
            }
        }

        // Draw original patch location in bright green
        self.draw_rectangle(
            &mut overlay,
            patch_location,
            original_patch.width(),
            original_patch.height(),
            Rgb([0, 255, 0]),
        );

        // Draw aligned location in red
        // Use the location from the result, which already contains the absolute position
        let aligned_x = alignment_result.location.x as u32;
        let aligned_y = alignment_result.location.y as u32;
        let aligned_width = alignment_result.location.width as u32;
        let aligned_height = alignment_result.location.height as u32;
        self.draw_rectangle(
            &mut overlay,
            (aligned_x, aligned_y),
            aligned_width,
            aligned_height,
            Rgb([255, 0, 0]),
        );

        overlay.save(output_path)?;
        Ok(())
    }

    fn draw_rectangle(
        &self,
        image: &mut RgbImage,
        location: (u32, u32),
        width: u32,
        height: u32,
        color: Rgb<u8>,
    ) {
        let (x, y) = location;
        let thickness = 3; // Make lines 3 pixels thick
        
        // Make colors brighter and more visible
        let bright_color = match color {
            Rgb([0, 255, 0]) => Rgb([50, 255, 50]), // Brighter green
            Rgb([255, 0, 0]) => Rgb([255, 50, 50]), // Brighter red
            _ => color,
        };

        // Draw thick horizontal lines (top and bottom)
        for t in 0..thickness {
            for i in 0..width {
                if x + i < image.width() {
                    // Top line
                    if y >= t && y - t < image.height() {
                        image.put_pixel(x + i, y - t, bright_color);
                    }
                    if y + t < image.height() {
                        image.put_pixel(x + i, y + t, bright_color);
                    }
                    // Bottom line
                    if y + height >= t && y + height - t < image.height() {
                        image.put_pixel(x + i, y + height - t, bright_color);
                    }
                    if y + height + t < image.height() {
                        image.put_pixel(x + i, y + height + t, bright_color);
                    }
                }
            }
        }

        // Draw thick vertical lines (left and right)
        for t in 0..thickness {
            for i in 0..height {
                if y + i < image.height() {
                    // Left line
                    if x >= t && x - t < image.width() {
                        image.put_pixel(x - t, y + i, bright_color);
                    }
                    if x + t < image.width() {
                        image.put_pixel(x + t, y + i, bright_color);
                    }
                    // Right line
                    if x + width >= t && x + width - t < image.width() {
                        image.put_pixel(x + width - t, y + i, bright_color);
                    }
                    if x + width + t < image.width() {
                        image.put_pixel(x + width + t, y + i, bright_color);
                    }
                }
            }
        }
    }

    fn create_error_heatmap(
        &self,
        original: &GrayImage,
        transformed: &GrayImage,
        _result: &AlignmentResult,
        output_path: &Path,
    ) -> crate::Result<()> {
        // Simple difference heatmap
        let (width, height) = (
            original.width().min(transformed.width()),
            original.height().min(transformed.height()),
        );
        let mut heatmap = RgbImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let orig_val = original.get_pixel(x, y)[0] as i32;
                let trans_val = transformed.get_pixel(x, y)[0] as i32;
                let diff = (orig_val - trans_val).unsigned_abs() as u8;

                // Red intensity proportional to difference
                heatmap.put_pixel(x, y, Rgb([diff, 0, 255 - diff]));
            }
        }

        heatmap.save(output_path)?;
        Ok(())
    }

    fn calculate_performance_metrics(
        &self,
        result: &AlignmentResult,
        patch_location: (u32, u32),
        patch_size: u32,
        _scenario: &TestScenario,
    ) -> PerformanceMetrics {
        // Calculate expected patch center (where we extracted the original patch)
        let expected_center_x = patch_location.0 as f32 + patch_size as f32 / 2.0;
        let expected_center_y = patch_location.1 as f32 + patch_size as f32 / 2.0;
        
        // Calculate actual detected center (where algorithm found the patch)
        let (actual_center_x, actual_center_y) = (
            result.location.x as f32 + result.location.width as f32 / 2.0,
            result.location.y as f32 + result.location.height as f32 / 2.0
        );
        
        // CORRECT translation error: distance between expected and actual centers
        let translation_error = ((actual_center_x - expected_center_x).powi(2) + 
                                (actual_center_y - expected_center_y).powi(2)).sqrt();

        // SUCCESS criteria: ONLY translation error and confidence
        let success_criteria = &self.config.validation.success_criteria;
        let success = translation_error < success_criteria.translation_accuracy_px
                     && result.confidence > success_criteria.min_confidence as f64;

        PerformanceMetrics {
            translation_error_px: translation_error,
            processing_time_ms: result.execution_time_ms as f32,
            confidence_score: result.confidence as f32,
            success,
        }
    }

    /// Generate summary report
    fn generate_summary_report(&self, reports: &[TestReport], session_id: &str) -> crate::Result<()> {
        // First generate the aggregated statistics
        self.generate_aggregated_statistics(reports)?;

        let summary_path = self.output_dir.join("SUMMARY_REPORT.md");
        let mut summary = String::new();

        summary.push_str("# 🔬 Comprehensive Algorithm Performance Report\n\n");
        summary.push_str(&format!(
            "**Generated:** {}\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        summary.push_str(&format!("**Session ID:** {}\n", session_id));
        summary.push_str(&format!("**Total Tests:** {}\n\n", reports.len()));
        summary.push_str(
            "**Note:** Aggregated statistics are available in `aggregated_statistics.json`\n\n",
        );

        // Group by algorithm
        let mut algorithms: std::collections::HashMap<String, Vec<&TestReport>> =
            std::collections::HashMap::new();
        for report in reports {
            algorithms
                .entry(report.algorithm_name.clone())
                .or_default()
                .push(report);
        }

        summary.push_str("## 📊 Algorithm Performance Summary\n\n");
        summary.push_str(
            "| Algorithm | Tests | Success Rate | Avg Time (ms) | Avg Translation Error (px) |\n",
        );
        summary.push_str(
            "|-----------|-------|--------------|---------------|---------------------------|\n",
        );

        for (algo_name, algo_reports) in &algorithms {
            let success_count = algo_reports
                .iter()
                .filter(|r| r.performance_metrics.success)
                .count();
            let success_rate = success_count as f32 / algo_reports.len() as f32 * 100.0;
            let avg_time: f32 = algo_reports
                .iter()
                .map(|r| r.performance_metrics.processing_time_ms)
                .sum::<f32>()
                / algo_reports.len() as f32;
            let avg_error: f32 = algo_reports
                .iter()
                .map(|r| r.performance_metrics.translation_error_px)
                .sum::<f32>()
                / algo_reports.len() as f32;

            summary.push_str(&format!(
                "| {} | {} | {:.1}% | {:.1} | {:.2} |\n",
                algo_name,
                algo_reports.len(),
                success_rate,
                avg_time,
                avg_error
            ));
        }

        summary.push_str("\n## 🎯 Detailed Results\n\n");

        for (algo_name, algo_reports) in algorithms {
            summary.push_str(&format!("### {}\n\n", algo_name));

            for report in algo_reports {
                summary.push_str(&format!("#### Test: {}\n", report.test_id));
                summary.push_str(&format!(
                    "- **Patch Size:** {}x{}\n",
                    report.patch_info.size.0, report.patch_info.size.1
                ));
                summary.push_str(&format!(
                    "- **Transformation:** {}\n",
                    report.transformation_applied.noise_parameters
                ));
                summary.push_str(&format!(
                    "- **Processing Time:** {:.2}ms\n",
                    report.performance_metrics.processing_time_ms
                ));
                summary.push_str(&format!(
                    "- **Translation Error:** {:.2}px\n",
                    report.performance_metrics.translation_error_px
                ));
                summary.push_str(&format!(
                    "- **Confidence:** {:.3}\n",
                    report.performance_metrics.confidence_score
                ));
                summary.push_str(&format!(
                    "- **Success:** {}\n",
                    if report.performance_metrics.success {
                        "✅"
                    } else {
                        "❌"
                    }
                ));
                summary.push('\n');
            }
        }

        std::fs::write(summary_path, summary)?;
        Ok(())
    }

    /// Generate aggregated statistics in JSON format
    fn generate_aggregated_statistics(&self, reports: &[TestReport]) -> crate::Result<()> {
        use serde::{Deserialize, Serialize};

        #[derive(Serialize, Deserialize)]
        struct AlgorithmStats {
            algorithm: String,
            total_tests: usize,
            avg_confidence: f32,
            avg_translation_error: f32,
            avg_time_ms: f32,
            min_error: f32,
            max_error: f32,
            success_count: usize,
            success_rate: f32,
            confidence_type: String,
            algorithm_type: String,
        }

        #[derive(Serialize, Deserialize)]
        struct AggregatedStats {
            generated_at: String,
            total_tests: usize,
            test_configuration: TestConfiguration,
            algorithm_stats: Vec<AlgorithmStats>,
            summary: Summary,
        }

        #[derive(Serialize, Deserialize)]
        struct TestConfiguration {
            patch_sizes: Vec<u32>,
            scenarios: Vec<String>,
        }

        #[derive(Serialize, Deserialize)]
        struct Summary {
            best_accuracy_algorithm: String,
            best_accuracy_error: f32,
            fastest_algorithm: String,
            fastest_time: f32,
            highest_success_rate_algorithm: String,
            highest_success_rate: f32,
        }

        // Group by algorithm
        let mut algorithms: std::collections::HashMap<String, Vec<&TestReport>> =
            std::collections::HashMap::new();
        for report in reports {
            algorithms
                .entry(report.algorithm_name.clone())
                .or_default()
                .push(report);
        }

        // Compute stats for each algorithm
        let mut algorithm_stats = Vec::new();
        for (algo_name, algo_reports) in &algorithms {
            let success_count = algo_reports
                .iter()
                .filter(|r| r.performance_metrics.success)
                .count();
            let errors: Vec<f32> = algo_reports
                .iter()
                .map(|r| r.performance_metrics.translation_error_px)
                .collect();
            let confidences: Vec<f32> = algo_reports
                .iter()
                .map(|r| r.performance_metrics.confidence_score)
                .collect();
            let times: Vec<f32> = algo_reports
                .iter()
                .map(|r| r.performance_metrics.processing_time_ms)
                .collect();

            let (algorithm_type, confidence_type) = if algo_name.contains("NCC")
                || algo_name.contains("SSD")
                || algo_name.contains("CCORR")
            {
                ("Template Matching", "Pattern Correlation")
            } else if algo_name.contains("SIFT")
                || algo_name.contains("ORB")
                || algo_name.contains("AKAZE")
            {
                ("Feature-based", "Feature Reliability")
            } else {
                ("Unknown", "Unknown")
            };

            algorithm_stats.push(AlgorithmStats {
                algorithm: algo_name.clone(),
                total_tests: algo_reports.len(),
                avg_confidence: confidences.iter().sum::<f32>() / confidences.len() as f32,
                avg_translation_error: errors.iter().sum::<f32>() / errors.len() as f32,
                avg_time_ms: times.iter().sum::<f32>() / times.len() as f32,
                min_error: errors.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                max_error: errors.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                success_count,
                success_rate: success_count as f32 / algo_reports.len() as f32 * 100.0,
                confidence_type: confidence_type.to_string(),
                algorithm_type: algorithm_type.to_string(),
            });
        }

        // Sort by average error for better readability
        algorithm_stats.sort_by(|a, b| {
            a.avg_translation_error
                .partial_cmp(&b.avg_translation_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find best performers (with error handling for empty stats)
        let (best_accuracy_algorithm, best_accuracy_error) = if algorithm_stats.is_empty() {
            ("No data".to_string(), 0.0)
        } else {
            let best_accuracy = algorithm_stats
                .iter()
                .min_by(|a, b| {
                    a.avg_translation_error
                        .partial_cmp(&b.avg_translation_error)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            (
                best_accuracy.algorithm.clone(),
                best_accuracy.avg_translation_error,
            )
        };

        let (fastest_algorithm, fastest_time) = if algorithm_stats.is_empty() {
            ("No data".to_string(), 0.0)
        } else {
            let fastest = algorithm_stats
                .iter()
                .min_by(|a, b| {
                    a.avg_time_ms
                        .partial_cmp(&b.avg_time_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            (fastest.algorithm.clone(), fastest.avg_time_ms)
        };

        let (highest_success_rate_algorithm, highest_success_rate) = if algorithm_stats.is_empty() {
            ("No data".to_string(), 0.0)
        } else {
            let highest_success = algorithm_stats
                .iter()
                .max_by(|a, b| {
                    a.success_rate
                        .partial_cmp(&b.success_rate)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            (
                highest_success.algorithm.clone(),
                highest_success.success_rate,
            )
        };

        // Extract unique patch sizes and scenarios from reports
        let mut patch_sizes = std::collections::HashSet::new();
        let mut scenarios = std::collections::HashSet::new();
        for report in reports {
            patch_sizes.insert(report.patch_info.size.0);
            scenarios.insert(report.transformation_applied.noise_parameters.clone());
        }

        let aggregated = AggregatedStats {
            generated_at: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            total_tests: reports.len(),
            test_configuration: TestConfiguration {
                patch_sizes: patch_sizes.into_iter().collect(),
                scenarios: scenarios.into_iter().collect(),
            },
            algorithm_stats,
            summary: Summary {
                best_accuracy_algorithm,
                best_accuracy_error,
                fastest_algorithm,
                fastest_time,
                highest_success_rate_algorithm,
                highest_success_rate,
            },
        };

        // Save to JSON file
        let stats_path = self.output_dir.join("aggregated_statistics.json");
        let json = serde_json::to_string_pretty(&aggregated)?;
        std::fs::write(stats_path, json)?;

        Ok(())
    }

}

#[derive(Debug, Clone)]
pub struct TestScenario {
    pub name: String,
    pub noise_type: NoiseType,
    pub rotation_deg: f32,
    pub translation: (i32, i32),
    pub scale_factor: f32,
}

#[derive(Debug, Clone)]
pub enum NoiseType {
    None,
    Gaussian { sigma: f32 },
    SaltPepper { density: f32 },
    GaussianBlur { sigma: f32 },
    Brightness { delta: i32 },
}
