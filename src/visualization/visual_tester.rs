use crate::{AlignmentResult, algorithms::*};
use image::{GrayImage, Rgb, RgbImage, Luma, imageops};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub test_id: String,
    pub algorithm_name: String,
    pub original_image_path: String,
    pub patch_info: PatchInfo,
    pub transformation_applied: TransformationInfo,
    pub alignment_result: AlignmentResult,
    pub visual_outputs: VisualOutputs,
    pub performance_metrics: PerformanceMetrics,
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
    pub rotation_error_deg: f32,
    pub scale_error_ratio: f32,
    pub processing_time_ms: f32,
    pub confidence_score: f32,
    pub success: bool,
}

pub struct VisualTester {
    pub output_dir: PathBuf,
    pub algorithms: Vec<Box<dyn AlignmentAlgorithm>>,
}

impl VisualTester {
    pub fn new(output_dir: PathBuf) -> Self {
        // Create output directory
        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");
        
        // Initialize all algorithms
        let mut algorithms: Vec<Box<dyn AlignmentAlgorithm>> = Vec::new();
        algorithms.push(Box::new(NormalizedCrossCorrelation));
        algorithms.push(Box::new(SumOfSquaredDifferences));
        algorithms.push(Box::new(IntegralImageMatcher));
        algorithms.push(Box::new(PhaseCorrelation));
        algorithms.push(Box::new(FastORB::new()));
        algorithms.push(Box::new(ImprovedORB::new()));
        
        Self {
            output_dir,
            algorithms,
        }
    }
    
    /// Run comprehensive visual tests on a SEM image
    pub fn run_comprehensive_test(&mut self, sem_image_path: &Path) -> crate::Result<Vec<TestReport>> {
        println!("🔬 Starting comprehensive visual test on: {}", sem_image_path.display());
        
        // Load the SEM image
        let sem_image = image::open(sem_image_path)?.to_luma8();
        println!("📸 Loaded SEM image: {}x{}", sem_image.width(), sem_image.height());
        
        // Extract multiple patches of different sizes
        let patch_sizes = vec![32, 64, 128];
        let mut all_reports = Vec::new();
        
        for &patch_size in &patch_sizes {
            println!("\n🔍 Testing with patch size: {}x{}", patch_size, patch_size);
            
            // Extract 3 patches of this size
            let patches = self.extract_good_patches(&sem_image, patch_size, 3)?;
            
            for (patch_idx, (patch, x, y)) in patches.into_iter().enumerate() {
                let test_base_id = format!("{}x{}_patch{}", patch_size, patch_size, patch_idx + 1);
                
                // Test different noise and transformation scenarios
                let test_scenarios = self.create_test_scenarios();
                
                for scenario in test_scenarios {
                    // Apply transformations and noise to the patch
                    let transformed_patch = self.apply_transformation_and_noise(&patch, &scenario)?;
                    
                    // Test all algorithms
                    for algorithm in &self.algorithms {
                        let test_id = format!("{}_{}_{}",
                            test_base_id,
                            scenario.name,
                            algorithm.name()
                        );
                        
                        println!("  🧪 Testing: {}", test_id);
                        
                        let report = self.run_single_test(
                            &test_id,
                            sem_image_path,
                            &sem_image,
                            &patch,
                            (x, y),
                            &transformed_patch,
                            &scenario,
                            algorithm.as_ref()
                        )?;
                        
                        all_reports.push(report);
                    }
                }
            }
        }
        
        // Generate summary report
        self.generate_summary_report(&all_reports)?;
        
        Ok(all_reports)
    }
    
    /// Extract good patches with sufficient texture
    fn extract_good_patches(&self, image: &GrayImage, patch_size: u32, count: u32) -> crate::Result<Vec<(GrayImage, u32, u32)>> {
        let mut patches = Vec::new();
        let margin = patch_size / 2;
        let min_variance = 100.0;
        
        for attempt in 0..100 {
            let x = margin + (attempt * 47) % (image.width() - patch_size - margin);
            let y = margin + (attempt * 37) % (image.height() - patch_size - margin);
            
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
            return Err(anyhow::anyhow!("No suitable patches found with sufficient texture"));
        }
        
        Ok(patches)
    }
    
    fn calculate_patch_variance(&self, patch: &GrayImage) -> f32 {
        let mean: f32 = patch.pixels().map(|p| p[0] as f32).sum::<f32>() / (patch.width() * patch.height()) as f32;
        let variance: f32 = patch.pixels()
            .map(|p| {
                let diff = p[0] as f32 - mean;
                diff * diff
            })
            .sum::<f32>() / (patch.width() * patch.height()) as f32;
        variance
    }
    
    /// Create test scenarios (noise, blur, brightness, etc.)
    fn create_test_scenarios(&self) -> Vec<TestScenario> {
        vec![
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
                translation: (5, 3),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "translation_10px".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 0.0,
                translation: (10, 7),
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
                noise_type: NoiseType::Gaussian { sigma: 10.0 },
                rotation_deg: 0.0,
                translation: (5, 5),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "salt_pepper".to_string(),
                noise_type: NoiseType::SaltPepper { density: 0.05 },
                rotation_deg: 0.0,
                translation: (3, 4),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "gaussian_blur".to_string(),
                noise_type: NoiseType::GaussianBlur { sigma: 1.0 },
                rotation_deg: 0.0,
                translation: (6, 2),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "brightness_change".to_string(),
                noise_type: NoiseType::Brightness { delta: 30 },
                rotation_deg: 0.0,
                translation: (4, 8),
                scale_factor: 1.0,
            },
            TestScenario {
                name: "scale_120".to_string(),
                noise_type: NoiseType::None,
                rotation_deg: 0.0,
                translation: (0, 0),
                scale_factor: 1.2,
            },
            TestScenario {
                name: "combined_complex".to_string(),
                noise_type: NoiseType::Gaussian { sigma: 5.0 },
                rotation_deg: 15.0,
                translation: (8, 6),
                scale_factor: 1.1,
            },
        ]
    }
    
    /// Apply transformation and noise to a patch
    fn apply_transformation_and_noise(&self, patch: &GrayImage, scenario: &TestScenario) -> crate::Result<GrayImage> {
        let mut result = patch.clone();
        
        // Apply scaling
        if (scenario.scale_factor - 1.0).abs() > 0.01 {
            let new_width = (patch.width() as f32 * scenario.scale_factor) as u32;
            let new_height = (patch.height() as f32 * scenario.scale_factor) as u32;
            result = imageops::resize(&result, new_width, new_height, imageops::FilterType::Gaussian);
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
        
        let interpolated = p11 * (1.0 - fx) * (1.0 - fy) +
                          p21 * fx * (1.0 - fy) +
                          p12 * (1.0 - fx) * fy +
                          p22 * fx * fy;
        
        interpolated.round() as u8
    }
    
    fn apply_noise(&self, image: &GrayImage, noise_type: &NoiseType) -> crate::Result<GrayImage> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut result = image.clone();
        
        match noise_type {
            NoiseType::None => {},
            NoiseType::Gaussian { sigma } => {
                for pixel in result.pixels_mut() {
                    let noise: f32 = rng.gen::<f32>() * 2.0 - 1.0; // -1 to 1
                    let noisy_val = pixel[0] as f32 + noise * sigma;
                    pixel[0] = noisy_val.clamp(0.0, 255.0) as u8;
                }
            },
            NoiseType::SaltPepper { density } => {
                for pixel in result.pixels_mut() {
                    if rng.gen::<f32>() < *density {
                        pixel[0] = if rng.gen::<bool>() { 255 } else { 0 };
                    }
                }
            },
            NoiseType::GaussianBlur { sigma } => {
                // Simple box blur approximation
                result = self.apply_box_blur(&result, (*sigma * 2.0) as u32)?;
            },
            NoiseType::Brightness { delta } => {
                for pixel in result.pixels_mut() {
                    let new_val = pixel[0] as i32 + delta;
                    pixel[0] = new_val.clamp(0, 255) as u8;
                }
            },
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
    fn run_single_test(
        &self,
        test_id: &str,
        sem_image_path: &Path,
        sem_image: &GrayImage,
        original_patch: &GrayImage,
        patch_location: (u32, u32),
        transformed_patch: &GrayImage,
        scenario: &TestScenario,
        algorithm: &dyn AlignmentAlgorithm
    ) -> crate::Result<TestReport> {
        // Create test-specific directory
        let test_dir = self.output_dir.join(test_id);
        std::fs::create_dir_all(&test_dir)?;
        
        // Save original patch
        let patch_path = test_dir.join("1_original_patch.png");
        original_patch.save(&patch_path)?;
        
        // Save transformed patch
        let transformed_patch_path = test_dir.join("2_transformed_patch.png");
        transformed_patch.save(&transformed_patch_path)?;
        
        // Run alignment
        let alignment_result = algorithm.align(original_patch, transformed_patch)?;
        
        // Generate visualizations
        let visual_outputs = self.generate_visualizations(
            &test_dir,
            sem_image,
            original_patch,
            transformed_patch,
            patch_location,
            &alignment_result,
            scenario
        )?;
        
        // Calculate performance metrics
        let performance_metrics = self.calculate_performance_metrics(&alignment_result, scenario);
        
        // Create test report
        let report = TestReport {
            test_id: test_id.to_string(),
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
        };
        
        // Save individual test report
        let report_path = test_dir.join("test_report.json");
        let report_json = serde_json::to_string_pretty(&report)?;
        std::fs::write(report_path, report_json)?;
        
        Ok(report)
    }
    
    /// Generate visual outputs for the test
    fn generate_visualizations(
        &self,
        test_dir: &Path,
        sem_image: &GrayImage,
        original_patch: &GrayImage,
        transformed_patch: &GrayImage,
        patch_location: (u32, u32),
        alignment_result: &AlignmentResult,
        _scenario: &TestScenario
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
            &overlay_path
        )?;
        
        // Create error heatmap
        let heatmap_path = test_dir.join("5_error_heatmap.png");
        self.create_error_heatmap(original_patch, transformed_patch, alignment_result, &heatmap_path)?;
        
        Ok(VisualOutputs {
            side_by_side_path: side_by_side_path.to_string_lossy().to_string(),
            overlay_result_path: overlay_path.to_string_lossy().to_string(),
            error_heatmap_path: heatmap_path.to_string_lossy().to_string(),
        })
    }
    
    fn create_side_by_side_image(&self, img1: &GrayImage, img2: &GrayImage, output_path: &Path) -> crate::Result<()> {
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
        transformed_patch: &GrayImage,
        patch_location: (u32, u32),
        alignment_result: &AlignmentResult,
        output_path: &Path
    ) -> crate::Result<()> {
        let mut overlay = RgbImage::new(sem_image.width(), sem_image.height());
        
        // Convert SEM image to RGB
        for y in 0..sem_image.height() {
            for x in 0..sem_image.width() {
                let gray_val = sem_image.get_pixel(x, y)[0];
                overlay.put_pixel(x, y, Rgb([gray_val, gray_val, gray_val]));
            }
        }
        
        // Draw original patch location in green
        self.draw_rectangle(&mut overlay, patch_location, original_patch.width(), original_patch.height(), Rgb([0, 255, 0]));
        
        // Draw aligned location in red
        let aligned_x = patch_location.0 as f32 + alignment_result.translation.0;
        let aligned_y = patch_location.1 as f32 + alignment_result.translation.1;
        self.draw_rectangle(&mut overlay, (aligned_x as u32, aligned_y as u32), transformed_patch.width(), transformed_patch.height(), Rgb([255, 0, 0]));
        
        overlay.save(output_path)?;
        Ok(())
    }
    
    fn draw_rectangle(&self, image: &mut RgbImage, location: (u32, u32), width: u32, height: u32, color: Rgb<u8>) {
        let (x, y) = location;
        
        // Draw horizontal lines
        for i in 0..width {
            if x + i < image.width() {
                if y < image.height() {
                    image.put_pixel(x + i, y, color);
                }
                if y + height < image.height() {
                    image.put_pixel(x + i, y + height, color);
                }
            }
        }
        
        // Draw vertical lines
        for i in 0..height {
            if y + i < image.height() {
                if x < image.width() {
                    image.put_pixel(x, y + i, color);
                }
                if x + width < image.width() {
                    image.put_pixel(x + width, y + i, color);
                }
            }
        }
    }
    
    fn create_error_heatmap(&self, original: &GrayImage, transformed: &GrayImage, _result: &AlignmentResult, output_path: &Path) -> crate::Result<()> {
        // Simple difference heatmap
        let (width, height) = (original.width().min(transformed.width()), original.height().min(transformed.height()));
        let mut heatmap = RgbImage::new(width, height);
        
        for y in 0..height {
            for x in 0..width {
                let orig_val = original.get_pixel(x, y)[0] as i32;
                let trans_val = transformed.get_pixel(x, y)[0] as i32;
                let diff = (orig_val - trans_val).abs() as u8;
                
                // Red intensity proportional to difference
                heatmap.put_pixel(x, y, Rgb([diff, 0, 255 - diff]));
            }
        }
        
        heatmap.save(output_path)?;
        Ok(())
    }
    
    fn calculate_performance_metrics(&self, result: &AlignmentResult, scenario: &TestScenario) -> PerformanceMetrics {
        let translation_error = if scenario.translation == (0, 0) {
            (result.translation.0.powi(2) + result.translation.1.powi(2)).sqrt()
        } else {
            let expected_x = scenario.translation.0 as f32;
            let expected_y = scenario.translation.1 as f32;
            ((result.translation.0 - expected_x).powi(2) + (result.translation.1 - expected_y).powi(2)).sqrt()
        };
        
        let rotation_error = (result.rotation - scenario.rotation_deg).abs();
        let scale_error = (result.scale - scenario.scale_factor).abs();
        
        PerformanceMetrics {
            translation_error_px: translation_error,
            rotation_error_deg: rotation_error,
            scale_error_ratio: scale_error,
            processing_time_ms: result.processing_time_ms,
            confidence_score: result.confidence,
            success: translation_error < 10.0 && rotation_error < 45.0 && result.confidence > 0.1,
        }
    }
    
    /// Generate summary report
    fn generate_summary_report(&self, reports: &[TestReport]) -> crate::Result<()> {
        let summary_path = self.output_dir.join("SUMMARY_REPORT.md");
        let mut summary = String::new();
        
        summary.push_str("# 🔬 Comprehensive Algorithm Performance Report\n\n");
        summary.push_str(&format!("**Generated:** {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        summary.push_str(&format!("**Total Tests:** {}\n\n", reports.len()));
        
        // Group by algorithm
        let mut algorithms = std::collections::HashMap::new();
        for report in reports {
            algorithms.entry(report.algorithm_name.clone()).or_insert_with(Vec::new).push(report);
        }
        
        summary.push_str("## 📊 Algorithm Performance Summary\n\n");
        summary.push_str("| Algorithm | Tests | Success Rate | Avg Time (ms) | Avg Translation Error (px) |\n");
        summary.push_str("|-----------|-------|--------------|---------------|---------------------------|\n");
        
        for (algo_name, algo_reports) in &algorithms {
            let success_count = algo_reports.iter().filter(|r| r.performance_metrics.success).count();
            let success_rate = success_count as f32 / algo_reports.len() as f32 * 100.0;
            let avg_time: f32 = algo_reports.iter().map(|r| r.performance_metrics.processing_time_ms).sum::<f32>() / algo_reports.len() as f32;
            let avg_error: f32 = algo_reports.iter().map(|r| r.performance_metrics.translation_error_px).sum::<f32>() / algo_reports.len() as f32;
            
            summary.push_str(&format!("| {} | {} | {:.1}% | {:.1} | {:.2} |\n", 
                algo_name, algo_reports.len(), success_rate, avg_time, avg_error));
        }
        
        summary.push_str("\n## 🎯 Detailed Results\n\n");
        
        for (algo_name, algo_reports) in algorithms {
            summary.push_str(&format!("### {}\n\n", algo_name));
            
            for report in algo_reports {
                summary.push_str(&format!("#### Test: {}\n", report.test_id));
                summary.push_str(&format!("- **Patch Size:** {}x{}\n", report.patch_info.size.0, report.patch_info.size.1));
                summary.push_str(&format!("- **Transformation:** {}\n", report.transformation_applied.noise_parameters));
                summary.push_str(&format!("- **Processing Time:** {:.2}ms\n", report.performance_metrics.processing_time_ms));
                summary.push_str(&format!("- **Translation Error:** {:.2}px\n", report.performance_metrics.translation_error_px));
                summary.push_str(&format!("- **Confidence:** {:.3}\n", report.performance_metrics.confidence_score));
                summary.push_str(&format!("- **Success:** {}\n", if report.performance_metrics.success { "✅" } else { "❌" }));
                summary.push_str("\n");
            }
        }
        
        std::fs::write(summary_path, summary)?;
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