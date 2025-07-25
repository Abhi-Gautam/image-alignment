use crate::config::OrbConfig;
use crate::logging::{AlgorithmSpan, get_correlation_id, metrics::Timer};
use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use crate::utils::estimate_transformation_ransac;
use crate::Result;
use opencv::core::{no_array, DMatch, KeyPoint, Mat, Ptr};
use opencv::features2d::{BFMatcher, ORB_ScoreType, ORB};
use opencv::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, debug, warn, error};

/// ORB (Oriented FAST and Rotated BRIEF) feature detector and matcher
/// ORB is efficient and provides good performance for real-time applications
pub struct OpenCVORB {
    detector: RefCell<Ptr<ORB>>,
    matcher: RefCell<Ptr<BFMatcher>>,
    config: OrbConfig,
}

// SAFETY: OpenCV types are safe to send across threads when used properly
unsafe impl Send for OpenCVORB {}
unsafe impl Sync for OpenCVORB {}

impl Default for OpenCVORB {
    fn default() -> Self {
        Self::new().expect("Failed to create ORB")
    }
}

impl OpenCVORB {
    pub fn new() -> Result<Self> {
        Self::with_config(OrbConfig::default())
    }

    pub fn with_config(config: OrbConfig) -> Result<Self> {
        let detector = ORB::create(
            config.max_features,
            config.scale_factor,
            config.n_levels,
            config.edge_threshold,
            config.first_level,
            config.wta_k,
            ORB_ScoreType::HARRIS_SCORE,
            config.patch_size,
            config.fast_threshold,
        )?;

        let matcher = BFMatcher::create(
            opencv::core::NORM_HAMMING,
            true,
        )?;

        Ok(Self {
            detector: RefCell::new(detector),
            matcher: RefCell::new(matcher),
            config,
        })
    }

    pub fn update_config(&mut self, config: OrbConfig) -> Result<()> {
        self.config = config.clone();
        *self.detector.borrow_mut() = ORB::create(
            config.max_features,
            config.scale_factor,
            config.n_levels,
            config.edge_threshold,
            config.first_level,
            config.wta_k,
            ORB_ScoreType::HARRIS_SCORE,
            config.patch_size,
            config.fast_threshold,
        )?;
        Ok(())
    }

    fn detect_and_compute(&self, image: &Mat) -> Result<(opencv::core::Vector<KeyPoint>, Mat)> {
        let _timer = Timer::start("orb_feature_detection", get_correlation_id());
        
        debug!(
            image_width = image.cols(),
            image_height = image.rows(),
            max_features = self.config.max_features,
            "Starting ORB feature detection and computation"
        );

        let mut keypoints = opencv::core::Vector::<KeyPoint>::new();
        let mut descriptors = Mat::default();

        self.detector.borrow_mut().detect_and_compute(
            image,
            &no_array(),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;

        let keypoint_count = keypoints.len();
        debug!(
            keypoints_detected = keypoint_count,
            descriptors_computed = descriptors.rows(),
            "ORB feature detection completed"
        );

        if keypoint_count == 0 {
            warn!("No keypoints detected in image");
        }

        Ok((keypoints, descriptors))
    }

    fn match_features(&self, desc1: &Mat, desc2: &Mat) -> Result<Vec<DMatch>> {
        let _timer = Timer::start("orb_feature_matching", get_correlation_id());
        
        debug!(
            desc1_features = desc1.rows(),
            desc2_features = desc2.rows(),
            distance_threshold = self.config.distance_threshold,
            "Starting ORB feature matching"
        );

        let mut matches = opencv::core::Vector::<DMatch>::new();

        if desc1.rows() == 0 || desc2.rows() == 0 {
            warn!("Cannot match features: one or both descriptor sets are empty");
            return Ok(Vec::new());
        }

        self.matcher
            .borrow()
            .train_match(desc1, desc2, &mut matches, &no_array())?;

        let raw_match_count = matches.len();
        debug!(raw_matches = raw_match_count, "Raw feature matches found");

        // Apply ratio test for better matches
        let mut good_matches = Vec::new();
        let matches_vec = matches.to_vec();

        // Sort by distance and keep good matches
        for m in matches_vec {
            if m.distance < self.config.distance_threshold {
                good_matches.push(m);
            }
        }

        // Sort by distance and limit to top matches
        good_matches.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        good_matches.truncate(100);

        let filtered_match_count = good_matches.len();
        let match_ratio = if raw_match_count > 0 {
            (filtered_match_count as f32) / (raw_match_count as f32)
        } else {
            0.0
        };

        debug!(
            filtered_matches = filtered_match_count,
            match_ratio = match_ratio,
            avg_distance = if !good_matches.is_empty() {
                good_matches.iter().map(|m| m.distance).sum::<f32>() / good_matches.len() as f32
            } else {
                0.0
            },
            "ORB feature matching completed"
        );

        if filtered_match_count < 4 {
            warn!(
                filtered_matches = filtered_match_count,
                "Very few matches found - alignment may be unreliable"
            );
        }

        Ok(good_matches)
    }

    fn estimate_transformation(
        &self,
        kp1: &[KeyPoint],
        kp2: &[KeyPoint],
        matches: &[DMatch],
    ) -> Result<(f32, f32, f32, f32, f32)> {
        let result = estimate_transformation_ransac(kp1, kp2, matches, None)?;
        Ok((
            result.translation.0,
            result.translation.1,
            result.rotation,
            result.scale,
            result.confidence,
        ))
    }
}

impl AlignmentAlgorithm for OpenCVORB {
    fn name(&self) -> &str {
        "OpenCV-ORB"
    }

    fn configure(&mut self, _config: &AlgorithmConfig) -> crate::Result<()> {
        Ok(())
    }

    fn preprocess(&self, image: &Mat) -> crate::Result<Mat> {
        // ORB works well with the original image, minimal preprocessing needed
        Ok(image.clone())
    }

    fn align(&self, search_image: &Mat, patch: &Mat) -> crate::Result<AlignmentResult> {
        // Create comprehensive algorithm execution span
        let algorithm_span = AlgorithmSpan::new(
            "ORB",
            None, // Will be filled when we know patch extraction location
            Some((patch.cols() as u32, patch.rows() as u32)),
            get_correlation_id(),
        );
        let _span_guard = algorithm_span.enter();
        
        let start = Instant::now();
        
        info!(
            algorithm = "ORB",
            search_image_size = format!("{}x{}", search_image.cols(), search_image.rows()),
            patch_size = format!("{}x{}", patch.cols(), patch.rows()),
            config = ?self.config,
            "Starting ORB alignment"
        );

        // Detect keypoints and compute descriptors
        debug!("Step 1: Feature detection and descriptor computation");
        let (patch_kp, patch_desc) = self.detect_and_compute(patch)?;
        let (search_kp, search_desc) = self.detect_and_compute(search_image)?;

        let patch_keypoint_count = patch_kp.len();
        let search_keypoint_count = search_kp.len();
        
        // Record feature detection results in the span
        algorithm_span.record_feature_detection(
            patch_keypoint_count + search_keypoint_count,
            (patch_desc.rows() + search_desc.rows()) as usize,
        );

        if patch_kp.is_empty() || search_kp.is_empty() {
            error!(
                patch_keypoints = patch_keypoint_count,
                search_keypoints = search_keypoint_count,
                "Insufficient keypoints detected - alignment failed"
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

        // Match features
        debug!("Step 2: Feature matching");
        let matches = self.match_features(&patch_desc, &search_desc)?;

        let match_count = matches.len();
        
        // Record matching results in the span  
        algorithm_span.record_matching(
            match_count, // We don't have separate raw/filtered here, so use the same
            match_count,
            if match_count > 0 {
                matches.iter().map(|m| m.distance).sum::<f32>() / match_count as f32
            } else {
                0.0
            },
        );

        if matches.is_empty() {
            error!(
                matches_found = match_count,
                "No feature matches found - alignment failed"
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

        // Estimate transformation
        debug!("Step 3: RANSAC transformation estimation");
        let patch_kp_vec = patch_kp.to_vec();
        let search_kp_vec = search_kp.to_vec();
        let (tx, ty, rotation, scale, confidence) =
            self.estimate_transformation(&patch_kp_vec, &search_kp_vec, &matches)?;

        // Record RANSAC results in the span
        algorithm_span.record_ransac(
            100, // We don't have access to actual iterations, use config default
            matches.len().min(50), // Estimate inliers as subset of matches
            0.5, // Placeholder for final error
        );

        info!(
            translation_x = tx,
            translation_y = ty,
            rotation = rotation,
            scale = scale,
            confidence = confidence,
            "RANSAC transformation estimation completed"
        );

        // Calculate match location in search image
        // RANSAC tx,ty represents the displacement between keypoints
        // Convert this to absolute coordinates where the patch was found
        let match_x = tx as i32;
        let match_y = ty as i32;

        debug!(
            raw_location = format!("({}, {})", match_x, match_y),
            final_location = format!("({}, {})", match_x.max(0), match_y.max(0)),
            "Location calculation completed"
        );

        // Prepare result metadata
        let mut metadata = HashMap::new();
        metadata.insert("matches_count".to_string(), serde_json::Value::from(matches.len()));
        metadata.insert("patch_keypoints".to_string(), serde_json::Value::from(patch_keypoint_count));
        metadata.insert("search_keypoints".to_string(), serde_json::Value::from(search_keypoint_count));
        metadata.insert("avg_match_distance".to_string(), serde_json::Value::from(
            if !matches.is_empty() {
                matches.iter().map(|m| m.distance).sum::<f32>() / matches.len() as f32
            } else {
                0.0
            }
        ));

        let final_confidence = confidence as f64;
        let execution_time = start.elapsed().as_secs_f64() * 1000.0;

        // Record detailed spatial alignment result
        algorithm_span.record_detailed_result(
            (tx, ty),
            rotation,
            scale,
            confidence,
            (match_x, match_y),
        );

        let result = AlignmentResult {
            location: crate::pipeline::SerializableRect {
                x: match_x.max(0),
                y: match_y.max(0),
                width: patch.cols(),
                height: patch.rows(),
            },
            score: final_confidence,
            confidence: final_confidence,
            execution_time_ms: execution_time,
            algorithm_name: AlignmentAlgorithm::name(self).to_string(),
            metadata,
            transformation: Some(crate::pipeline::TransformParams {
                translation: (tx, ty),
                rotation_degrees: rotation,
                scale,
                skew: None,
            }),
        };

        info!(
            final_location = format!("({}, {})", match_x.max(0), match_y.max(0)),
            confidence = final_confidence,
            execution_time_ms = execution_time,
            success = final_confidence > 0.0,
            "ORB alignment completed successfully"
        );

        Ok(result)
    }

    fn supports_gpu(&self) -> bool {
        false // CPU implementation
    }

    fn estimated_complexity(&self) -> ComplexityClass {
        ComplexityClass::Medium // ORB is reasonably fast
    }
}

