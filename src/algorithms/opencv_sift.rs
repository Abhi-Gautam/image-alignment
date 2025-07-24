use crate::config::SiftConfig;
use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use crate::utils::estimate_transformation_ransac;
use crate::Result;
use opencv::core::{no_array, DMatch, KeyPoint, Mat, Ptr};
use opencv::features2d::{BFMatcher, SIFT};
use opencv::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

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
            "SIFT RANSAC result: tx={}, ty={}, rotation={}, confidence={}",
            tx,
            ty,
            rotation,
            confidence
        );

        // Calculate match location in search image
        // RANSAC tx,ty represents the displacement between keypoints
        let match_x = tx as i32;
        let match_y = ty as i32;

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
        ComplexityClass::High // SIFT is computationally expensive but very robust
    }
}

