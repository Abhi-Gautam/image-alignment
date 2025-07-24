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
    n_features: i32,
    n_octave_layers: i32,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_features: usize,
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
        let detector = SIFT::create(
            0,     // nfeatures (0 = no limit)
            3,     // nOctaveLayers
            0.04,  // contrastThreshold
            10.0,  // edgeThreshold
            1.6,   // sigma
            false, // enable_precise_upscale
        )?;

        let matcher = BFMatcher::create(
            opencv::core::NORM_L2, // norm type for float descriptors
            true,                  // cross check
        )?;

        Ok(Self {
            detector: RefCell::new(detector),
            matcher: RefCell::new(matcher),
            n_features: 0,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
            max_features: 1000,
        })
    }

    pub fn with_n_features(mut self, n_features: i32) -> Result<Self> {
        self.n_features = n_features;
        *self.detector.borrow_mut() = SIFT::create(
            n_features,
            self.n_octave_layers,
            self.contrast_threshold,
            self.edge_threshold,
            self.sigma,
            false,
        )?;
        Ok(self)
    }

    pub fn with_contrast_threshold(mut self, threshold: f64) -> Result<Self> {
        self.contrast_threshold = threshold;
        *self.detector.borrow_mut() = SIFT::create(
            self.n_features,
            self.n_octave_layers,
            threshold,
            self.edge_threshold,
            self.sigma,
            false,
        )?;
        Ok(self)
    }

    pub fn with_edge_threshold(mut self, threshold: f64) -> Result<Self> {
        self.edge_threshold = threshold;
        *self.detector.borrow_mut() = SIFT::create(
            self.n_features,
            self.n_octave_layers,
            self.contrast_threshold,
            threshold,
            self.sigma,
            false,
        )?;
        Ok(self)
    }

    pub fn with_sigma(mut self, sigma: f64) -> Result<Self> {
        self.sigma = sigma;
        *self.detector.borrow_mut() = SIFT::create(
            self.n_features,
            self.n_octave_layers,
            self.contrast_threshold,
            self.edge_threshold,
            sigma,
            false,
        )?;
        Ok(self)
    }

    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
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

        // Limit number of features if specified
        if keypoints.len() > self.max_features {
            // Sort by response (strength) and keep the best
            let mut kp_vec: Vec<KeyPoint> = keypoints.to_vec();
            kp_vec.sort_by(|a, b| {
                b.response()
                    .partial_cmp(&a.response())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            kp_vec.truncate(self.max_features);

            keypoints = opencv::core::Vector::from_iter(kp_vec);

            // Extract corresponding descriptors
            let mut limited_descriptors = Mat::zeros(
                self.max_features as i32,
                descriptors.cols(),
                descriptors.typ(),
            )?
            .to_mat()?;

            for i in 0..self.max_features.min(descriptors.rows() as usize) {
                let src_row = descriptors.row(i as i32)?;
                let dst_row = limited_descriptors.row_mut(i as i32)?;
                src_row.copy_to(&mut dst_row.clone_pointee())?;
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
        let distance_threshold = 100.0; // Adjust this threshold as needed
        good_matches.retain(|m| m.distance < distance_threshold);
        good_matches.truncate(100); // Limit to top 100 matches

        Ok(good_matches)
    }

    fn estimate_transformation(
        &self,
        kp1: &[KeyPoint],
        kp2: &[KeyPoint],
        matches: &[DMatch],
    ) -> Result<(f32, f32, f32, f32)> {
        let result = estimate_transformation_ransac(kp1, kp2, matches, None)?;
        Ok((result.translation.0, result.translation.1, result.rotation, result.confidence))
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
        let (tx, ty, rotation, confidence) =
            self.estimate_transformation(&patch_kp_vec, &search_kp_vec, &matches)?;

        // Calculate match location in search image
        let match_x = (patch.cols() / 2) + tx as i32;
        let match_y = (patch.rows() / 2) + ty as i32;

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
            serde_json::Value::from(self.contrast_threshold),
        );
        metadata.insert(
            "edge_threshold".to_string(),
            serde_json::Value::from(self.edge_threshold),
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
                scale: 1.0, // Will be updated in future versions
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_test_pattern(width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            // Create a pattern with distinct features for SIFT detection
            let intensity = if (x % 20 < 10) ^ (y % 20 < 10) {
                if (x % 40 < 20) ^ (y % 40 < 20) {
                    255
                } else {
                    192
                }
            } else {
                if (x % 40 < 20) ^ (y % 40 < 20) {
                    128
                } else {
                    64
                }
            };
            Luma([intensity])
        })
    }


    #[test]
    fn test_sift_creation() {
        let sift = OpenCVSIFT::new();
        assert!(sift.is_ok());
    }

    #[test]
    fn test_sift_with_params() {
        let sift_result = OpenCVSIFT::new()
            .unwrap()
            .with_contrast_threshold(0.03)
            .unwrap()
            .with_max_features(500);

        assert_eq!(sift_result.max_features, 500);
    }

    #[test]
    fn test_sift_alignment() {
        let sift = OpenCVSIFT::new().unwrap();
        let template = create_test_pattern(64, 64);
        let target = create_test_pattern(64, 64);

        let template_mat = crate::utils::image_conversion::grayimage_to_mat(&template).unwrap();
        let target_mat = crate::utils::image_conversion::grayimage_to_mat(&target).unwrap();

        let result = sift.align(&target_mat, &template_mat);
        match &result {
            Ok(_) => {}
            Err(e) => {
                eprintln!("SIFT alignment failed with error: {}", e);
                eprintln!("Error details: {:?}", e);
            }
        }
        assert!(result.is_ok());

        let alignment = result.unwrap();
        assert_eq!(alignment.algorithm_name, "OpenCV-SIFT");
        assert!(alignment.execution_time_ms >= 0.0);
        assert!(alignment.confidence >= 0.0 && alignment.confidence <= 1.0);
    }

    #[test]
    fn test_grayimage_to_mat_conversion() {
        let image = create_test_pattern(32, 32);

        let mat_result = crate::utils::image_conversion::grayimage_to_mat(&image);
        assert!(mat_result.is_ok());

        let mat = mat_result.unwrap();
        assert_eq!(mat.cols(), 32);
        assert_eq!(mat.rows(), 32);
        assert_eq!(mat.channels(), 1);
    }
}
