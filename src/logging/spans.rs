//! Structured spans for hierarchical logging
//!
//! Provides pre-defined span structures for different components of the alignment system,
//! ensuring consistent logging patterns and correlation tracking.

use std::time::Instant;
use tracing::{span, Level, Span};
use uuid::Uuid;

/// Span for algorithm execution with detailed context
pub struct AlgorithmSpan {
    span: Span,
    start_time: Instant,
    algorithm_name: String,
    patch_location: Option<(u32, u32)>,
    patch_size: Option<(u32, u32)>,
}

impl AlgorithmSpan {
    /// Create a new algorithm execution span
    pub fn new(
        algorithm_name: &str,
        patch_location: Option<(u32, u32)>,
        patch_size: Option<(u32, u32)>,
        correlation_id: Option<Uuid>,
    ) -> Self {
        // Try to get patch context from thread-local storage if not provided
        let (final_patch_location, final_patch_size) = if let Some(context) = crate::logging::get_patch_context() {
            (
                patch_location.or(Some(context.extracted_from)),
                patch_size.or(Some(context.patch_size)),
            )
        } else {
            (patch_location, patch_size)
        };
        let span = if let Some(corr_id) = correlation_id {
            span!(
                Level::INFO,
                "algorithm_execution",
                algorithm = algorithm_name,
                patch_x = final_patch_location.map(|(x, _)| x),
                patch_y = final_patch_location.map(|(_, y)| y),
                patch_width = final_patch_size.map(|(w, _)| w),
                patch_height = final_patch_size.map(|(_, h)| h),
                correlation_id = %corr_id
            )
        } else {
            span!(
                Level::INFO,
                "algorithm_execution",
                algorithm = algorithm_name,
                patch_x = final_patch_location.map(|(x, _)| x),
                patch_y = final_patch_location.map(|(_, y)| y),
                patch_width = final_patch_size.map(|(w, _)| w),
                patch_height = final_patch_size.map(|(_, h)| h)
            )
        };

        Self {
            span,
            start_time: Instant::now(),
            algorithm_name: algorithm_name.to_string(),
            patch_location: final_patch_location,
            patch_size: final_patch_size,
        }
    }

    /// Record feature detection results
    pub fn record_feature_detection(&self, keypoints: usize, descriptors: usize) {
        self.span.record("keypoints_detected", keypoints);
        self.span.record("descriptors_computed", descriptors);
        tracing::debug!(
            parent: &self.span,
            keypoints = keypoints,
            descriptors = descriptors,
            "Feature detection completed"
        );
    }

    /// Record matching results
    pub fn record_matching(&self, raw_matches: usize, filtered_matches: usize, match_confidence: f32) {
        self.span.record("raw_matches", raw_matches);
        self.span.record("filtered_matches", filtered_matches);
        self.span.record("match_confidence", match_confidence);
        tracing::debug!(
            parent: &self.span,
            raw_matches = raw_matches,
            filtered_matches = filtered_matches,
            match_confidence = match_confidence,
            "Feature matching completed"
        );
    }

    /// Record RANSAC estimation results
    pub fn record_ransac(&self, iterations: usize, inliers: usize, final_error: f32) {
        self.span.record("ransac_iterations", iterations);
        self.span.record("ransac_inliers", inliers);
        self.span.record("ransac_error", final_error);
        tracing::debug!(
            parent: &self.span,
            iterations = iterations,
            inliers = inliers,
            final_error = final_error,
            "RANSAC estimation completed"
        );
    }

    /// Record final alignment result with success/failure status
    pub fn record_result(&self, success: bool, confidence: f64, description: &str) {
        let duration = self.start_time.elapsed();
        self.span.record("success", success);
        self.span.record("final_confidence", confidence);
        self.span.record("execution_time_ms", duration.as_millis() as f64);
        self.span.record("result_description", description);
        
        tracing::info!(
            parent: &self.span,
            success = success,
            confidence = confidence,
            execution_time_ms = duration.as_millis(),
            description = description,
            "Algorithm execution completed"
        );
    }

    /// Record detailed alignment result with spatial information
    pub fn record_detailed_result(
        &self, 
        translation: (f32, f32), 
        rotation: f32, 
        scale: f32, 
        confidence: f32,
        found_location: (i32, i32),
    ) {
        let duration = self.start_time.elapsed();
        self.span.record("translation_x", translation.0);
        self.span.record("translation_y", translation.1);
        self.span.record("rotation", rotation);
        self.span.record("scale", scale);
        self.span.record("final_confidence", confidence);
        self.span.record("found_x", found_location.0);
        self.span.record("found_y", found_location.1);
        self.span.record("execution_time_ms", duration.as_millis() as f64);
        
        // Calculate spatial accuracy if we have patch location
        if let Some((orig_x, orig_y)) = self.patch_location {
            let distance = (((found_location.0 - orig_x as i32).pow(2) + 
                            (found_location.1 - orig_y as i32).pow(2)) as f32).sqrt();
            let translation_error = (translation.0.powi(2) + translation.1.powi(2)).sqrt();
            
            self.span.record("patch_to_match_distance", distance);
            self.span.record("translation_error", translation_error);
            
            tracing::info!(
                parent: &self.span,
                patch_extracted_at = format!("({}, {})", orig_x, orig_y),
                match_found_at = format!("({}, {})", found_location.0, found_location.1),
                spatial_distance = format!("{:.2}px", distance),
                translation_error = format!("{:.2}px", translation_error),
                translation = format!("({:.2}, {:.2})", translation.0, translation.1),
                rotation = format!("{:.2}°", rotation),
                scale = format!("{:.3}x", scale),
                confidence = format!("{:.3}", confidence),
                execution_time_ms = duration.as_millis(),
                "Detailed spatial alignment result recorded"
            );
        } else {
            tracing::info!(
                parent: &self.span,
                match_found_at = format!("({}, {})", found_location.0, found_location.1),
                translation = format!("({:.2}, {:.2})", translation.0, translation.1),
                rotation = format!("{:.2}°", rotation),
                scale = format!("{:.3}x", scale),
                confidence = format!("{:.3}", confidence),
                execution_time_ms = duration.as_millis(),
                "Alignment result recorded (patch origin unknown)"
            );
        }
    }

    /// Record patch extraction context
    pub fn record_patch_context(&self, extracted_from: (u32, u32), variance: f32, source_image_size: (u32, u32)) {
        self.span.record("patch_extracted_x", extracted_from.0);
        self.span.record("patch_extracted_y", extracted_from.1);
        self.span.record("patch_variance", variance);
        self.span.record("source_width", source_image_size.0);
        self.span.record("source_height", source_image_size.1);
        
        tracing::debug!(
            parent: &self.span,
            patch_extracted_at = format!("({}, {})", extracted_from.0, extracted_from.1),
            patch_variance = format!("{:.1}", variance),
            source_image_size = format!("{}x{}", source_image_size.0, source_image_size.1),
            "Patch extraction context recorded"
        );
    }

    /// Get the underlying span for manual instrumentation
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Enter the span context
    pub fn enter(&self) -> tracing::span::Entered<'_> {
        self.span.enter()
    }
}

/// Span for pipeline stage execution
pub struct PipelineSpan {
    span: Span,
    start_time: Instant,
    stage_name: String,
}

impl PipelineSpan {
    /// Create a new pipeline stage span
    pub fn new(stage_name: &str, correlation_id: Option<Uuid>) -> Self {
        let span = if let Some(corr_id) = correlation_id {
            span!(
                Level::INFO,
                "pipeline_stage",
                stage = stage_name,
                correlation_id = %corr_id
            )
        } else {
            span!(
                Level::INFO,
                "pipeline_stage",
                stage = stage_name
            )
        };

        Self {
            span,
            start_time: Instant::now(),
            stage_name: stage_name.to_string(),
        }
    }

    /// Record stage input metadata
    pub fn record_input(&self, input_type: &str, input_size: Option<(u32, u32)>) {
        self.span.record("input_type", input_type);
        if let Some((width, height)) = input_size {
            self.span.record("input_width", width);
            self.span.record("input_height", height);
        }
        tracing::debug!(
            parent: &self.span,
            input_type = input_type,
            input_width = input_size.map(|(w, _)| w),
            input_height = input_size.map(|(_, h)| h),
            "Pipeline stage input recorded"
        );
    }

    /// Record stage completion
    pub fn record_completion(&self, output_type: &str, success: bool) {
        let duration = self.start_time.elapsed();
        self.span.record("output_type", output_type);
        self.span.record("success", success);
        self.span.record("execution_time_ms", duration.as_millis() as f64);
        
        tracing::info!(
            parent: &self.span,
            output_type = output_type,
            success = success,
            execution_time_ms = duration.as_millis(),
            "Pipeline stage completed"
        );
    }

    /// Get the underlying span
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Enter the span context
    pub fn enter(&self) -> tracing::span::Entered<'_> {
        self.span.enter()
    }
}

/// Span for test session tracking
pub struct TestSessionSpan {
    span: Span,
    start_time: Instant,
    session_id: Uuid,
    test_type: String,
}

impl TestSessionSpan {
    /// Create a new test session span
    pub fn new(test_type: &str, session_id: Uuid) -> Self {
        let span = span!(
            Level::INFO,
            "test_session",
            test_type = test_type,
            session_id = %session_id
        );

        Self {
            span,
            start_time: Instant::now(),
            session_id,
            test_type: test_type.to_string(),
        }
    }

    /// Record test configuration
    pub fn record_config(&self, config: &str, parameters: serde_json::Value) {
        self.span.record("test_config", config);
        tracing::info!(
            parent: &self.span,
            config = config,
            parameters = %parameters,
            "Test session configuration recorded"
        );
    }

    /// Record test iteration
    pub fn record_iteration(&self, iteration: usize, algorithm: &str, success: bool) {
        tracing::debug!(
            parent: &self.span,
            iteration = iteration,
            algorithm = algorithm,
            success = success,
            "Test iteration completed"
        );
    }

    /// Record session completion
    pub fn record_completion(&self, total_tests: usize, successful_tests: usize) {
        let duration = self.start_time.elapsed();
        let success_rate = if total_tests > 0 {
            (successful_tests as f64) / (total_tests as f64) * 100.0
        } else {
            0.0
        };

        self.span.record("total_tests", total_tests);
        self.span.record("successful_tests", successful_tests);
        self.span.record("success_rate", success_rate);
        self.span.record("session_duration_ms", duration.as_millis() as f64);
        
        tracing::info!(
            parent: &self.span,
            total_tests = total_tests,
            successful_tests = successful_tests,
            success_rate = success_rate,
            session_duration_ms = duration.as_millis(),
            "Test session completed"
        );
    }

    /// Get the session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Get the underlying span
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Enter the span context
    pub fn enter(&self) -> tracing::span::Entered<'_> {
        self.span.enter()
    }
}

/// Create a simple span with correlation ID using dynamic name
pub fn create_info_span(name: &str, correlation_id: Option<Uuid>) -> Span {
    if let Some(corr_id) = correlation_id {
        tracing::info_span!("dynamic_span", name = name, correlation_id = %corr_id)
    } else {
        tracing::info_span!("dynamic_span", name = name)
    }
}

/// Create a debug span with correlation ID using dynamic name
pub fn create_debug_span(name: &str, correlation_id: Option<Uuid>) -> Span {
    if let Some(corr_id) = correlation_id {
        tracing::debug_span!("dynamic_span", name = name, correlation_id = %corr_id)
    } else {
        tracing::debug_span!("dynamic_span", name = name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn test_algorithm_span() {
        let correlation_id = Uuid::new_v4();
        let span = AlgorithmSpan::new(
            "ORB",
            Some((100, 200)),
            Some((64, 64)),
            Some(correlation_id),
        );

        let _enter = span.enter();
        span.record_feature_detection(150, 128);
        span.record_matching(75, 45, 0.85);
        span.record_ransac(100, 35, 0.25);
        span.record_result((12.5, -8.2), 15.3, 1.02, 0.89);
    }

    #[traced_test]
    #[test]
    fn test_pipeline_span() {
        let correlation_id = Uuid::new_v4();
        let span = PipelineSpan::new("preprocessing", Some(correlation_id));

        let _enter = span.enter();
        span.record_input("raw_image", Some((1024, 768)));
        span.record_completion("processed_image", true);
    }

    #[traced_test]
    #[test]
    fn test_session_span() {
        let session_id = Uuid::new_v4();
        let span = TestSessionSpan::new("visual_test", session_id);

        let _enter = span.enter();
        let config = serde_json::json!({
            "algorithm": "ORB",
            "patch_size": 64,
            "iterations": 10
        });
        span.record_config("default", config);
        span.record_iteration(1, "ORB", true);
        span.record_iteration(2, "SIFT", false);
        span.record_completion(10, 8);
    }
}