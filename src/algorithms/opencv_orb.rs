use crate::config::OrbConfig;
use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use crate::utils::estimate_transformation_ransac;
use crate::Result;
use opencv::core::{no_array, DMatch, KeyPoint, Mat, Ptr};
use opencv::features2d::{BFMatcher, ORB_ScoreType, ORB};
use opencv::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

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
        let mut keypoints = opencv::core::Vector::<KeyPoint>::new();
        let mut descriptors = Mat::default();

        self.detector.borrow_mut().detect_and_compute(
            image,
            &no_array(),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;

        Ok((keypoints, descriptors))
    }

    fn match_features(&self, desc1: &Mat, desc2: &Mat) -> Result<Vec<DMatch>> {
        let mut matches = opencv::core::Vector::<DMatch>::new();

        if desc1.rows() == 0 || desc2.rows() == 0 {
            return Ok(Vec::new());
        }

        self.matcher
            .borrow()
            .train_match(desc1, desc2, &mut matches, &no_array())?;

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
        let start = Instant::now();

        // Detect keypoints and compute descriptors
        let (patch_kp, patch_desc) = self.detect_and_compute(patch)?;
        let (search_kp, search_desc) = self.detect_and_compute(search_image)?;

        if patch_kp.is_empty() || search_kp.is_empty() {
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
        let matches = self.match_features(&patch_desc, &search_desc)?;

        if matches.is_empty() {
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
        let patch_kp_vec = patch_kp.to_vec();
        let search_kp_vec = search_kp.to_vec();
        let (tx, ty, rotation, scale, confidence) =
            self.estimate_transformation(&patch_kp_vec, &search_kp_vec, &matches)?;

        // Debug RANSAC output
        log::info!(
            "ORB RANSAC result: tx={}, ty={}, rotation={}, scale={}, confidence={}",
            tx,
            ty,
            rotation,
            scale,
            confidence
        );
        log::info!(
            "Patch size: {}x{}, Search size: {}x{}",
            patch.cols(),
            patch.rows(),
            search_image.cols(),
            search_image.rows()
        );

        // Calculate match location in search image
        // RANSAC tx,ty represents the displacement between keypoints
        // Convert this to absolute coordinates where the patch was found
        // The displacement should tell us where the patch top-left corner is located
        let match_x = tx as i32;
        let match_y = ty as i32;

        log::info!("ORB calculated match location: ({}, {})", match_x, match_y);
        log::info!(
            "ORB final location after max(0): ({}, {})",
            match_x.max(0),
            match_y.max(0)
        );

        let mut metadata = HashMap::new();
        metadata.insert(
            "matches_count".to_string(),
            serde_json::Value::from(matches.len()),
        );
        metadata.insert(
            "patch_keypoints".to_string(),
            serde_json::Value::from(patch_kp.len()),
        );
        metadata.insert(
            "search_keypoints".to_string(),
            serde_json::Value::from(search_kp.len()),
        );

        Ok(AlignmentResult {
            location: crate::pipeline::SerializableRect {
                x: match_x.max(0),
                y: match_y.max(0),
                width: patch.cols(),
                height: patch.rows(),
            },
            score: confidence as f64,
            confidence: confidence as f64,
            execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            algorithm_name: AlignmentAlgorithm::name(self).to_string(),
            metadata,
            transformation: Some(crate::pipeline::TransformParams {
                translation: (tx, ty),
                rotation_degrees: rotation,
                scale,
                skew: None,
            }),
        })
    }

    fn supports_gpu(&self) -> bool {
        false // CPU implementation
    }

    fn estimated_complexity(&self) -> ComplexityClass {
        ComplexityClass::Medium // ORB is reasonably fast
    }
}

