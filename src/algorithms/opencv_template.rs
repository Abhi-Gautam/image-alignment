use crate::{AlignmentResult, Result};
use crate::algorithms::AlignmentAlgorithm;
use image::GrayImage;
use instant::Instant;

/// OpenCV-based template matching algorithms
/// This is a mock implementation that simulates OpenCV's matchTemplate functionality
/// When OpenCV is properly installed, this will use cv::matchTemplate()

#[derive(Debug)]
pub enum TemplateMatchMode {
    /// Normalized Cross Correlation (equivalent to cv::TM_CCOEFF_NORMED)
    NormalizedCrossCorrelation,
    /// Sum of Squared Differences (equivalent to cv::TM_SQDIFF_NORMED)
    SumOfSquaredDifferences,
    /// Correlation Coefficient (equivalent to cv::TM_CCORR_NORMED)
    CorrelationCoefficient,
}

pub struct OpenCVTemplateMatcher {
    mode: TemplateMatchMode,
}

impl OpenCVTemplateMatcher {
    pub fn new(mode: TemplateMatchMode) -> Self {
        Self { mode }
    }

    pub fn new_ncc() -> Self {
        Self::new(TemplateMatchMode::NormalizedCrossCorrelation)
    }

    pub fn new_ssd() -> Self {
        Self::new(TemplateMatchMode::SumOfSquaredDifferences)
    }

    pub fn new_ccorr() -> Self {
        Self::new(TemplateMatchMode::CorrelationCoefficient)
    }

    /// Mock implementation of OpenCV's matchTemplate
    /// TODO: Replace with actual cv::matchTemplate when OpenCV is available
    fn match_template(&self, target: &GrayImage, template: &GrayImage) -> Result<(f32, f32, f64)> {
        let target_width = target.width() as i32;
        let target_height = target.height() as i32;
        let template_width = template.width() as i32;
        let template_height = template.height() as i32;

        if template_width > target_width || template_height > target_height {
            return Err(anyhow::anyhow!("Template larger than target image"));
        }

        let mut best_match_x = 0i32;
        let mut best_match_y = 0i32;
        let mut best_score = match self.mode {
            TemplateMatchMode::SumOfSquaredDifferences => f64::MAX,
            _ => f64::MIN,
        };

        // Sliding window template matching
        for y in 0..=(target_height - template_height) {
            for x in 0..=(target_width - template_width) {
                let score = self.calculate_match_score(target, template, x, y)?;
                
                let is_better = match self.mode {
                    TemplateMatchMode::SumOfSquaredDifferences => score < best_score,
                    _ => score > best_score,
                };

                if is_better {
                    best_score = score;
                    best_match_x = x;
                    best_match_y = y;
                }
            }
        }

        // Calculate center position
        let center_x = best_match_x as f32 + template_width as f32 / 2.0;
        let center_y = best_match_y as f32 + template_height as f32 / 2.0;

        Ok((center_x, center_y, best_score))
    }

    fn calculate_match_score(&self, target: &GrayImage, template: &GrayImage, offset_x: i32, offset_y: i32) -> Result<f64> {
        let template_width = template.width() as i32;
        let template_height = template.height() as i32;

        match self.mode {
            TemplateMatchMode::NormalizedCrossCorrelation => {
                self.compute_ncc(target, template, offset_x, offset_y)
            }
            TemplateMatchMode::SumOfSquaredDifferences => {
                self.compute_ssd(target, template, offset_x, offset_y)
            }
            TemplateMatchMode::CorrelationCoefficient => {
                self.compute_ccorr(target, template, offset_x, offset_y)
            }
        }
    }

    fn compute_ncc(&self, target: &GrayImage, template: &GrayImage, offset_x: i32, offset_y: i32) -> Result<f64> {
        let template_width = template.width() as i32;
        let template_height = template.height() as i32;

        let mut target_sum = 0.0;
        let mut template_sum = 0.0;
        let mut target_sq_sum = 0.0;
        let mut template_sq_sum = 0.0;
        let mut cross_sum = 0.0;
        let n = (template_width * template_height) as f64;

        for ty in 0..template_height {
            for tx in 0..template_width {
                let target_x = (offset_x + tx) as u32;
                let target_y = (offset_y + ty) as u32;
                
                let target_pixel = target.get_pixel(target_x, target_y)[0] as f64;
                let template_pixel = template.get_pixel(tx as u32, ty as u32)[0] as f64;

                target_sum += target_pixel;
                template_sum += template_pixel;
                target_sq_sum += target_pixel * target_pixel;
                template_sq_sum += template_pixel * template_pixel;
                cross_sum += target_pixel * template_pixel;
            }
        }

        let target_mean = target_sum / n;
        let template_mean = template_sum / n;

        let numerator = cross_sum - n * target_mean * template_mean;
        let target_var = target_sq_sum - n * target_mean * target_mean;
        let template_var = template_sq_sum - n * template_mean * template_mean;
        let denominator = (target_var * template_var).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn compute_ssd(&self, target: &GrayImage, template: &GrayImage, offset_x: i32, offset_y: i32) -> Result<f64> {
        let template_width = template.width() as i32;
        let template_height = template.height() as i32;

        let mut ssd = 0.0;
        let mut template_sq_sum = 0.0;

        for ty in 0..template_height {
            for tx in 0..template_width {
                let target_x = (offset_x + tx) as u32;
                let target_y = (offset_y + ty) as u32;
                
                let target_pixel = target.get_pixel(target_x, target_y)[0] as f64;
                let template_pixel = template.get_pixel(tx as u32, ty as u32)[0] as f64;

                let diff = target_pixel - template_pixel;
                ssd += diff * diff;
                template_sq_sum += template_pixel * template_pixel;
            }
        }

        // Normalize by template energy
        if template_sq_sum == 0.0 {
            Ok(f64::MAX)
        } else {
            Ok(ssd / template_sq_sum)
        }
    }

    fn compute_ccorr(&self, target: &GrayImage, template: &GrayImage, offset_x: i32, offset_y: i32) -> Result<f64> {
        let template_width = template.width() as i32;
        let template_height = template.height() as i32;

        let mut target_sum = 0.0;
        let mut template_sum = 0.0;
        let mut cross_sum = 0.0;

        for ty in 0..template_height {
            for tx in 0..template_width {
                let target_x = (offset_x + tx) as u32;
                let target_y = (offset_y + ty) as u32;
                
                let target_pixel = target.get_pixel(target_x, target_y)[0] as f64;
                let template_pixel = template.get_pixel(tx as u32, ty as u32)[0] as f64;

                target_sum += target_pixel;
                template_sum += template_pixel;
                cross_sum += target_pixel * template_pixel;
            }
        }

        let n = (template_width * template_height) as f64;
        
        if template_sum == 0.0 || target_sum == 0.0 {
            Ok(0.0)
        } else {
            Ok(cross_sum / (target_sum * template_sum).sqrt())
        }
    }
}

impl AlignmentAlgorithm for OpenCVTemplateMatcher {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> Result<AlignmentResult> {
        let start = Instant::now();

        let (x, y, confidence) = self.match_template(target, template)?;

        // Calculate translation from center of target image
        let target_center_x = target.width() as f32 / 2.0;
        let target_center_y = target.height() as f32 / 2.0;
        
        let translation_x = x - target_center_x;
        let translation_y = y - target_center_y;

        let processing_time = start.elapsed().as_secs_f32() * 1000.0;

        // Normalize confidence score to [0, 1] range
        let normalized_confidence = match self.mode {
            TemplateMatchMode::SumOfSquaredDifferences => {
                // For SSD, lower is better, so invert and normalize
                (1.0 / (1.0 + confidence)).max(0.0).min(1.0) as f32
            }
            _ => {
                // For correlation methods, higher is better
                confidence.max(0.0).min(1.0) as f32
            }
        };

        Ok(AlignmentResult {
            translation: (translation_x, translation_y),
            rotation: 0.0, // Template matching doesn't detect rotation
            scale: 1.0,     // Template matching doesn't detect scale
            confidence: normalized_confidence,
            processing_time_ms: processing_time,
            algorithm_used: self.name().to_string(),
        })
    }

    fn name(&self) -> &'static str {
        match self.mode {
            TemplateMatchMode::NormalizedCrossCorrelation => "OpenCV-NCC",
            TemplateMatchMode::SumOfSquaredDifferences => "OpenCV-SSD",
            TemplateMatchMode::CorrelationCoefficient => "OpenCV-CCORR",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_test_pattern(width: u32, height: u32, pattern_type: u8) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            match pattern_type {
                0 => Luma([((x + y) % 2 * 255) as u8]), // Checkerboard
                1 => Luma([((x % 8 < 4) ^ (y % 8 < 4)) as u8 * 255]), // Grid
                _ => Luma([128]), // Gray
            }
        })
    }

    #[test]
    fn test_ncc_exact_match() {
        let template = create_test_pattern(16, 16, 0);
        let target = create_test_pattern(16, 16, 0);
        
        let matcher = OpenCVTemplateMatcher::new_ncc();
        let result = matcher.align(&template, &target).unwrap();
        
        assert_eq!(result.algorithm_used, "OpenCV-NCC");
        assert!((result.translation.0.abs() < 1.0), "Translation X should be near 0, got {}", result.translation.0);
        assert!((result.translation.1.abs() < 1.0), "Translation Y should be near 0, got {}", result.translation.1);
        assert!(result.confidence > 0.8, "Confidence should be high for exact match, got {}", result.confidence);
    }

    #[test]
    fn test_ssd_exact_match() {
        let template = create_test_pattern(16, 16, 1);
        let target = create_test_pattern(16, 16, 1);
        
        let matcher = OpenCVTemplateMatcher::new_ssd();
        let result = matcher.align(&template, &target).unwrap();
        
        assert_eq!(result.algorithm_used, "OpenCV-SSD");
        assert!((result.translation.0.abs() < 1.0), "Translation should be near 0");
        assert!((result.translation.1.abs() < 1.0), "Translation should be near 0");
        assert!(result.confidence > 0.5, "Confidence should be reasonable for exact match");
    }

    #[test]
    fn test_template_larger_than_target() {
        let template = create_test_pattern(32, 32, 0);
        let target = create_test_pattern(16, 16, 0);
        
        let matcher = OpenCVTemplateMatcher::new_ncc();
        let result = matcher.align(&template, &target);
        
        assert!(result.is_err(), "Should fail when template is larger than target");
    }
}