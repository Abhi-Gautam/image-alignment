use crate::config::SiftConfig;
use crate::logging::{AlgorithmSpan, get_correlation_id, metrics::Timer};
use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use crate::utils::estimate_transformation_ransac;
use crate::Result;
use opencv::core::{no_array, DMatch, KeyPoint, Mat, Ptr};
use opencv::features2d::{BFMatcher, SIFT};
use opencv::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, debug, warn};

/// SIFT (Scale-Invariant Feature Transform) feature detector and matcher
/// SIFT is highly robust to scale, rotation, and illumination changes
pub struct OpenCVSIFT {
    detector: RefCell<Ptr<SIFT>>,
    matcher: RefCell<Ptr<BFMatcher>>,
    config: SiftConfig,
}

// SAFETY: OpenCV types are safe to send across threads when used properly
unsafe impl Send for OpenCVSIFT {}
unsafe impl Sync for OpenCVSIFT {}

impl Default for OpenCVSIFT {
    fn default() -> Self {
        Self::new().expect("Failed to create SIFT")
    }
}

impl OpenCVSIFT {
    pub fn new() -> Result<Self> {
        Self::with_config(SiftConfig::default())
    }

    pub fn with_config(config: SiftConfig) -> Result<Self> {
        let detector = SIFT::create(
            config.n_features,
            config.n_octave_layers,
            config.contrast_threshold,
            config.edge_threshold,
            config.sigma,
            false, // enable_precise_upscale
        )?;

        let matcher = BFMatcher::create(
            opencv::core::NORM_L2, // norm type for float descriptors
            true,                  // cross check
        )?;

        Ok(Self {
            detector: RefCell::new(detector),
            matcher: RefCell::new(matcher),
            config,
        })
    }

    pub fn update_config(&mut self, config: SiftConfig) -> Result<()> {
        self.config = config.clone();
        *self.detector.borrow_mut() = SIFT::create(
            config.n_features,
            config.n_octave_layers,
            config.contrast_threshold,
            config.edge_threshold,
            config.sigma,
            false,
        )?;
        Ok(())
    }

    pub fn with_n_features(mut self, n_features: i32) -> Result<Self> {
        self.config.n_features = n_features;
        self.update_config(self.config.clone())?;
        Ok(self)
    }

    pub fn with_contrast_threshold(mut self, threshold: f64) -> Result<Self> {
        self.config.contrast_threshold = threshold;
        self.update_config(self.config.clone())?;
        Ok(self)
    }

    pub fn with_edge_threshold(mut self, threshold: f64) -> Result<Self> {
        self.config.edge_threshold = threshold;
        self.update_config(self.config.clone())?;
        Ok(self)
    }

    pub fn with_sigma(mut self, sigma: f64) -> Result<Self> {
        self.config.sigma = sigma;
        self.update_config(self.config.clone())?;
        Ok(self)
    }

    pub fn with_max_features(mut self, max_features: usize) -> Self {
        // Note: SIFT config doesn't have max_features, but we can use n_features instead
        self.config.n_features = max_features as i32;
        self
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

        // Limit number of features if specified (use n_features as max limit)
        let max_features = if self.config.n_features > 0 {
            self.config.n_features as usize
        } else {
            1000 // default limit if n_features is 0 (unlimited)
        };
        
        if keypoints.len() > max_features {
            // Create indexed pairs to maintain keypoint-descriptor correspondence
            let kp_vec: Vec<KeyPoint> = keypoints.to_vec();
            let mut kp_with_indices: Vec<(usize, KeyPoint)> =
                kp_vec.into_iter().enumerate().collect();

            // Sort by response (strength) and keep the best
            kp_with_indices.sort_by(|a, b| {
                b.1.response()
                    .partial_cmp(&a.1.response())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            kp_with_indices.truncate(max_features);

            // Extract sorted keypoints
            let sorted_keypoints: Vec<KeyPoint> =
                kp_with_indices.iter().map(|(_, kp)| kp.clone()).collect();
            keypoints = opencv::core::Vector::from_iter(sorted_keypoints);

            // Extract corresponding descriptors in the same order
            let mut limited_descriptors = Mat::zeros(
                max_features as i32,
                descriptors.cols(),
                descriptors.typ(),
            )?
            .to_mat()?;

            for (dst_idx, (src_idx, _)) in kp_with_indices.iter().enumerate() {
                if *src_idx < descriptors.rows() as usize {
                    let src_row = descriptors.row(*src_idx as i32)?;
                    let dst_row = limited_descriptors.row_mut(dst_idx as i32)?;
                    src_row.copy_to(&mut dst_row.clone_pointee())?;
                }
            }

            descriptors = limited_descriptors;
        }

        Ok((keypoints, descriptors))
    }

    fn match_features(&self, desc1: &Mat, desc2: &Mat) -> Result<Vec<DMatch>> {
        let mut matches = opencv::core::Vector::<DMatch>::new();

        if desc1.rows() == 0 || desc2.rows() == 0 {
            return Ok(Vec::new());
        }

        // Use simple match instead of knnMatch to avoid the batch distance issue
        self.matcher
            .borrow_mut()
            .train_match(desc1, desc2, &mut matches, &no_array())?;

        // Filter matches by distance threshold since we're using simple matching
        let mut good_matches = matches.to_vec();

        // Sort by distance and keep the best matches
        good_matches.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep matches with distance below threshold
        good_matches.retain(|m| m.distance < self.config.distance_threshold);
        good_matches.truncate(100); // Limit to top 100 matches

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

impl AlignmentAlgorithm for OpenCVSIFT {
    fn name(&self) -> &str {
        "OpenCV-SIFT"
    }

    fn configure(&mut self, config: &AlgorithmConfig) -> crate::Result<()> {
        if let Some(n_features) = config.parameters.get("n_features").and_then(|v| v.as_i64()) {
            *self = std::mem::take(self).with_n_features(n_features as i32)?;
        }

        if let Some(contrast_threshold) = config
            .parameters
            .get("contrast_threshold")
            .and_then(|v| v.as_f64())
        {
            *self = std::mem::take(self).with_contrast_threshold(contrast_threshold)?;
        }

        if let Some(edge_threshold) = config
            .parameters
            .get("edge_threshold")
            .and_then(|v| v.as_f64())
        {
            *self = std::mem::take(self).with_edge_threshold(edge_threshold)?;
        }

        if let Some(sigma) = config.parameters.get("sigma").and_then(|v| v.as_f64()) {
            *self = std::mem::take(self).with_sigma(sigma)?;
        }

        if let Some(max_features) = config
            .parameters
            .get("max_features")
            .and_then(|v| v.as_u64())
        {
            *self = std::mem::take(self).with_max_features(max_features as usize);
        }

        Ok(())
    }

    fn preprocess(&self, image: &Mat) -> crate::Result<Mat> {
        // SIFT works well with the original image, minimal preprocessing needed
        Ok(image.clone())
    }

    fn align(&self, search_image: &Mat, patch: &Mat) -> crate::Result<AlignmentResult> {
        // Create comprehensive algorithm execution span
        let algorithm_span = AlgorithmSpan::new(
            "SIFT",
            None,
            Some((patch.cols() as u32, patch.rows() as u32)),
            get_correlation_id(),
        );
        let _span_guard = algorithm_span.enter();
        
        let start = Instant::now();
        
        info!(
            algorithm = "SIFT",
            search_image_size = format!("{}x{}", search_image.cols(), search_image.rows()),
            patch_size = format!("{}x{}", patch.cols(), patch.rows()),
            n_features = self.config.n_features,
            contrast_threshold = self.config.contrast_threshold,
            edge_threshold = self.config.edge_threshold,
            sigma = self.config.sigma,
            "Starting SIFT feature-based alignment"
        );

        // Step 1: Feature detection and descriptor computation
        debug!("Step 1: Detecting SIFT features and computing descriptors");
        let _detection_timer = Timer::start("sift_feature_detection", get_correlation_id());
        
        let (patch_kp, patch_desc) = self.detect_and_compute(patch)?;
        let (search_kp, search_desc) = self.detect_and_compute(search_image)?;
        
        // Record feature detection results
        algorithm_span.record_feature_detection(
            patch_kp.len() + search_kp.len(),
            (patch_desc.rows() + search_desc.rows()) as usize,
        );
        
        info!(
            patch_keypoints = patch_kp.len(),
            search_keypoints = search_kp.len(),
            patch_descriptors = patch_desc.rows(),
            search_descriptors = search_desc.rows(),
            "SIFT feature detection completed"
        );

        if patch_kp.is_empty() || search_kp.is_empty() {
            warn!(
                patch_features = patch_kp.len(),
                search_features = search_kp.len(),
                "Insufficient SIFT features detected - alignment failed"
            );
            
            algorithm_span.record_result(false, 0.0, "No features detected");
            
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

        // Step 2: Feature matching
        debug!("Step 2: Matching SIFT features between patch and search image");
        let _matching_timer = Timer::start("sift_feature_matching", get_correlation_id());
        
        let matches = self.match_features(&patch_desc, &search_desc)?;
        
        // Record matching results
        algorithm_span.record_matching(
            matches.len(),
            matches.len(), // For SIFT, all raw matches are good matches after filtering  
            if matches.is_empty() { 0.0 } else { 1.0 - (matches.iter().map(|m| m.distance as f32).sum::<f32>() / matches.len() as f32) / 256.0 },
        );
        
        info!(
            raw_matches = matches.len(),
            distance_threshold = self.config.distance_threshold,
            "SIFT feature matching completed"
        );

        if matches.is_empty() {
            warn!("No valid SIFT feature matches found - alignment failed");
            
            algorithm_span.record_result(false, 0.0, "No matches found");
            
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

        // Step 3: Geometric transformation estimation with RANSAC
        debug!("Step 3: Estimating geometric transformation using RANSAC");
        let _ransac_timer = Timer::start("sift_ransac_estimation", get_correlation_id());
        
        let patch_kp_vec = patch_kp.to_vec();
        let search_kp_vec = search_kp.to_vec();
        let (tx, ty, rotation, scale, confidence) =
            self.estimate_transformation(&patch_kp_vec, &search_kp_vec, &matches)?;

        info!(
            translation = format!("({:.2}, {:.2})", tx, ty),
            rotation_degrees = format!("{:.2}°", rotation),
            scale = format!("{:.3}x", scale),
            confidence = format!("{:.3}", confidence),
            "SIFT RANSAC transformation estimation completed"
        );

        // Calculate match location in search image
        // RANSAC tx,ty represents the displacement between keypoints
        let match_x = tx as i32;
        let match_y = ty as i32;

        // Record detailed spatial alignment result
        algorithm_span.record_detailed_result(
            (tx, ty),
            rotation,
            scale,
            confidence,
            (match_x, match_y),
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
        metadata.insert(
            "contrast_threshold".to_string(),
            serde_json::Value::from(self.config.contrast_threshold),
        );
        metadata.insert(
            "edge_threshold".to_string(),
            serde_json::Value::from(self.config.edge_threshold),
        );
        metadata.insert(
            "n_features".to_string(),
            serde_json::Value::from(self.config.n_features),
        );
        metadata.insert(
            "sigma".to_string(),
            serde_json::Value::from(self.config.sigma),
        );

        let execution_time = start.elapsed().as_secs_f64() * 1000.0;
        
        info!(
            execution_time_ms = format!("{:.2}", execution_time),
            final_confidence = format!("{:.3}", confidence),
            match_location = format!("({}, {})", match_x, match_y),
            "SIFT alignment completed successfully"
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
            execution_time_ms: execution_time,
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
        ComplexityClass::High // SIFT is computationally expensive but very robust
    }
}

