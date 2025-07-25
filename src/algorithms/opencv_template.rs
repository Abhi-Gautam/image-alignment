use crate::config::TemplateConfig;
use crate::logging::{AlgorithmSpan, get_correlation_id, metrics::Timer};
use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use opencv::core::{no_array, Mat, Point2i};
use opencv::imgproc;
use opencv::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, debug, warn, error};

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
    config: TemplateConfig,
}

impl OpenCVTemplateMatcher {
    pub fn new(mode: TemplateMatchMode) -> Self {
        Self { 
            mode,
            config: TemplateConfig::default(),
        }
    }

    pub fn with_config(mode: TemplateMatchMode, config: TemplateConfig) -> Self {
        Self { mode, config }
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
        // Create comprehensive algorithm execution span
        let algorithm_name = match self.mode {
            TemplateMatchMode::NormalizedCrossCorrelation => "Template-NCC",
            TemplateMatchMode::SumOfSquaredDifferences => "Template-SSD", 
            TemplateMatchMode::CorrelationCoefficient => "Template-CCORR",
        };
        
        let algorithm_span = AlgorithmSpan::new(
            algorithm_name,
            None,
            Some((patch.cols() as u32, patch.rows() as u32)),
            get_correlation_id(),
        );
        let _span_guard = algorithm_span.enter();
        
        let start = Instant::now();
        
        info!(
            algorithm = algorithm_name,
            search_image_size = format!("{}x{}", search_image.cols(), search_image.rows()),
            patch_size = format!("{}x{}", patch.cols(), patch.rows()),
            mode = ?self.mode,
            "Starting template matching alignment"
        );

        if patch.cols() > search_image.cols() || patch.rows() > search_image.rows() {
            error!(
                patch_size = format!("{}x{}", patch.cols(), patch.rows()),
                search_size = format!("{}x{}", search_image.cols(), search_image.rows()),
                "Patch larger than search image - alignment impossible"
            );
            
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

        // Record template matching as "feature detection" stage
        debug!("Step 1: Performing template matching");
        let _timer = Timer::start("template_matching", get_correlation_id());
        
        let mut result = Mat::default();
        imgproc::match_template(
            search_image,
            patch,
            &mut result,
            self.mode_to_opencv(),
            &no_array(),
        )?;

        // Record the template matching process
        algorithm_span.record_feature_detection(
            1, // Template matching produces 1 "feature" (the match location)
            result.rows() as usize * result.cols() as usize, // Number of comparison points
        );

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
                let normalized = (1.0 / (1.0 + score)).clamp(0.0, 1.0);
                if normalized < self.config.confidence_threshold as f64 {
                    0.0
                } else {
                    normalized
                }
            }
            _ => {
                // For correlation methods, higher is better
                let normalized = score.clamp(0.0, 1.0);
                if normalized < self.config.confidence_threshold as f64 {
                    0.0
                } else {
                    normalized
                }
            }
        };

        // Record detailed spatial alignment result
        algorithm_span.record_detailed_result(
            (best_loc.x as f32, best_loc.y as f32),
            0.0,  // Template matching doesn't detect rotation
            1.0,  // Template matching doesn't detect scale
            confidence as f32,
            (best_loc.x, best_loc.y),
        );

        let mut metadata = HashMap::new();
        metadata.insert(
            "match_method".to_string(),
            serde_json::Value::from(format!("{:?}", self.mode)),
        );
        metadata.insert("raw_score".to_string(), serde_json::Value::from(score));

        info!(
            final_location = format!("({}, {})", best_loc.x, best_loc.y),
            confidence = format!("{:.3}", confidence),
            raw_score = format!("{:.3}", score),
            "Template matching alignment completed successfully"
        );

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

