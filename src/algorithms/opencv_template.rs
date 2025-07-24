use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use opencv::core::{no_array, Mat, Point2i};
use opencv::imgproc;
use opencv::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// OpenCV-based template matching algorithms
/// Uses real OpenCV's matchTemplate functionality
#[derive(Debug, Clone, Copy)]
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

// SAFETY: OpenCV template matching is safe to send across threads
unsafe impl Send for OpenCVTemplateMatcher {}
unsafe impl Sync for OpenCVTemplateMatcher {}

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

    fn mode_to_opencv(&self) -> i32 {
        match self.mode {
            TemplateMatchMode::NormalizedCrossCorrelation => imgproc::TM_CCOEFF_NORMED,
            TemplateMatchMode::SumOfSquaredDifferences => imgproc::TM_SQDIFF_NORMED,
            TemplateMatchMode::CorrelationCoefficient => imgproc::TM_CCORR_NORMED,
        }
    }
}

impl AlignmentAlgorithm for OpenCVTemplateMatcher {
    fn name(&self) -> &str {
        match self.mode {
            TemplateMatchMode::NormalizedCrossCorrelation => "OpenCV-NCC",
            TemplateMatchMode::SumOfSquaredDifferences => "OpenCV-SSD",
            TemplateMatchMode::CorrelationCoefficient => "OpenCV-CCORR",
        }
    }

    fn configure(&mut self, config: &AlgorithmConfig) -> crate::Result<()> {
        if let Some(mode_str) = config.parameters.get("mode").and_then(|v| v.as_str()) {
            self.mode = match mode_str {
                "ncc" | "normalized_cross_correlation" => {
                    TemplateMatchMode::NormalizedCrossCorrelation
                }
                "ssd" | "sum_of_squared_differences" => TemplateMatchMode::SumOfSquaredDifferences,
                "ccorr" | "correlation_coefficient" => TemplateMatchMode::CorrelationCoefficient,
                _ => self.mode, // Keep current if unknown
            };
        }

        Ok(())
    }

    fn preprocess(&self, image: &Mat) -> crate::Result<Mat> {
        // Template matching works well with the original image
        Ok(image.clone())
    }

    fn align(&self, search_image: &Mat, patch: &Mat) -> crate::Result<AlignmentResult> {
        let start = Instant::now();

        if patch.cols() > search_image.cols() || patch.rows() > search_image.rows() {
            return Ok(AlignmentResult {
                location: crate::pipeline::SerializableRect {
                    x: 0,
                    y: 0,
                    width: patch.cols(),
                    height: patch.rows(),
                },
                score: 0.0,
                confidence: 0.0,
                execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                algorithm_name: AlignmentAlgorithm::name(self).to_string(),
                metadata: HashMap::new(),
                transformation: None,
            });
        }

        // Perform template matching
        let mut result = Mat::default();
        imgproc::match_template(
            search_image,
            patch,
            &mut result,
            self.mode_to_opencv(),
            &no_array(),
        )?;

        // Find the best match location
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = Point2i::default();
        let mut max_loc = Point2i::default();

        opencv::core::min_max_loc(
            &result,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &no_array(),
        )?;

        // Choose the appropriate location and score based on method
        let (best_loc, score) = match self.mode {
            TemplateMatchMode::SumOfSquaredDifferences => {
                // For SQDIFF, smaller values mean better matches
                (min_loc, min_val)
            }
            _ => {
                // For other methods, larger values mean better matches
                (max_loc, max_val)
            }
        };

        // Normalize confidence score
        let confidence = match self.mode {
            TemplateMatchMode::SumOfSquaredDifferences => {
                // For SSD, lower is better, so invert and normalize
                (1.0 / (1.0 + score)).clamp(0.0, 1.0)
            }
            _ => {
                // For correlation methods, higher is better
                score.clamp(0.0, 1.0)
            }
        };

        let mut metadata = HashMap::new();
        metadata.insert(
            "match_method".to_string(),
            serde_json::Value::from(format!("{:?}", self.mode)),
        );
        metadata.insert("raw_score".to_string(), serde_json::Value::from(score));

        Ok(AlignmentResult {
            location: crate::pipeline::SerializableRect {
                x: best_loc.x,
                y: best_loc.y,
                width: patch.cols(),
                height: patch.rows(),
            },
            score: confidence,
            confidence,
            execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            algorithm_name: AlignmentAlgorithm::name(self).to_string(),
            metadata,
            transformation: Some(crate::pipeline::TransformParams {
                translation: (best_loc.x as f32, best_loc.y as f32),
                rotation_degrees: 0.0,
                scale: 1.0,
                skew: None,
            }),
        })
    }

    fn supports_gpu(&self) -> bool {
        false // CPU implementation
    }

    fn estimated_complexity(&self) -> ComplexityClass {
        ComplexityClass::Low // Template matching is relatively fast
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::grayimage_to_mat;
    use image::{GrayImage, Luma};

    fn create_test_pattern(width: u32, height: u32, pattern_type: u8) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            match pattern_type {
                0 => Luma([((x + y) % 2 * 255) as u8]), // Checkerboard
                1 => Luma([((x % 8 < 4) ^ (y % 8 < 4)) as u8 * 255]), // Grid
                _ => Luma([128]),                       // Gray
            }
        })
    }


    #[test]
    fn test_ncc_exact_match() {
        let template = create_test_pattern(16, 16, 0);
        let target = create_test_pattern(16, 16, 0);

        let template_mat = grayimage_to_mat(&template).unwrap();
        let target_mat = grayimage_to_mat(&target).unwrap();

        let matcher = OpenCVTemplateMatcher::new_ncc();
        let result = matcher.align(&target_mat, &template_mat).unwrap();

        assert_eq!(result.algorithm_name, "OpenCV-NCC");
        assert!(
            result.transformation.is_some(),
            "Transformation should be present"
        );
        let transform = result.transformation.unwrap();
        assert!(
            (transform.translation.0.abs() < 1.0),
            "Translation X should be near 0, got {}",
            transform.translation.0
        );
        assert!(
            (transform.translation.1.abs() < 1.0),
            "Translation Y should be near 0, got {}",
            transform.translation.1
        );
        assert!(
            result.confidence > 0.8,
            "Confidence should be high for exact match, got {}",
            result.confidence
        );
        assert!(
            result.execution_time_ms >= 0.0,
            "Execution time should be non-negative"
        );
    }

    #[test]
    fn test_ssd_exact_match() {
        let template = create_test_pattern(16, 16, 1);
        let target = create_test_pattern(16, 16, 1);

        let template_mat = grayimage_to_mat(&template).unwrap();
        let target_mat = grayimage_to_mat(&target).unwrap();

        let matcher = OpenCVTemplateMatcher::new_ssd();
        let result = matcher.align(&target_mat, &template_mat).unwrap();

        assert_eq!(result.algorithm_name, "OpenCV-SSD");
        assert!(
            result.transformation.is_some(),
            "Transformation should be present"
        );
        let transform = result.transformation.unwrap();
        assert!(
            (transform.translation.0.abs() < 1.0),
            "Translation should be near 0"
        );
        assert!(
            (transform.translation.1.abs() < 1.0),
            "Translation should be near 0"
        );
        assert!(
            result.confidence > 0.5,
            "Confidence should be reasonable for exact match"
        );
        assert!(
            result.execution_time_ms >= 0.0,
            "Execution time should be non-negative"
        );
    }

    #[test]
    fn test_template_larger_than_target() {
        let template = create_test_pattern(32, 32, 0);
        let target = create_test_pattern(16, 16, 0);

        let template_mat = grayimage_to_mat(&template).unwrap();
        let target_mat = grayimage_to_mat(&target).unwrap();

        let matcher = OpenCVTemplateMatcher::new_ncc();
        let result = matcher.align(&target_mat, &template_mat).unwrap();

        // Should return low confidence result rather than error
        assert_eq!(result.algorithm_name, "OpenCV-NCC");
        assert_eq!(
            result.confidence, 0.0,
            "Should have zero confidence when template is larger"
        );
        assert!(
            result.execution_time_ms >= 0.0,
            "Execution time should be non-negative"
        );
    }

    #[test]
    fn test_grayimage_to_mat_conversion() {
        let image = create_test_pattern(32, 32, 0);

        let mat_result = grayimage_to_mat(&image);
        assert!(mat_result.is_ok());

        let mat = mat_result.unwrap();
        assert_eq!(mat.cols(), 32);
        assert_eq!(mat.rows(), 32);
        assert_eq!(mat.channels(), 1);
    }
}
