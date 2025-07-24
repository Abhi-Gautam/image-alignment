use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use crate::utils::estimate_transformation_ransac;
use crate::Result;
use opencv::core::{no_array, DMatch, KeyPoint, Mat, Ptr};
use opencv::features2d::{AKAZE_DescriptorType, BFMatcher, KAZE_DiffusivityType, AKAZE};
use opencv::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

/// AKAZE (Accelerated-KAZE) feature detector and matcher
/// AKAZE is robust to rotation and scale changes, making it ideal for alignment tasks
pub struct OpenCVAKAZE {
    detector: RefCell<Ptr<AKAZE>>,
    matcher: RefCell<Ptr<BFMatcher>>,
    threshold: f32,
    octaves: i32,
    octave_layers: i32,
    diffusivity: KAZE_DiffusivityType,
    max_features: usize,
}

// SAFETY: OpenCV types are safe to send across threads when used properly
unsafe impl Send for OpenCVAKAZE {}
unsafe impl Sync for OpenCVAKAZE {}

impl Default for OpenCVAKAZE {
    fn default() -> Self {
        Self::new().expect("Failed to create AKAZE")
    }
}

impl OpenCVAKAZE {
    pub fn new() -> Result<Self> {
        let detector = AKAZE::create(
            AKAZE_DescriptorType::DESCRIPTOR_MLDB,
            0,        // descriptor size (0 = full)
            3,        // descriptor channels
            0.001f32, // threshold
            4,        // octaves
            4,        // octave layers
            KAZE_DiffusivityType::DIFF_PM_G2,
            -1, // max_points (-1 = no limit)
        )?;

        let matcher = BFMatcher::create(
            opencv::core::NORM_HAMMING, // norm type for binary descriptors
            true,                       // cross check
        )?;

        Ok(Self {
            detector: RefCell::new(detector),
            matcher: RefCell::new(matcher),
            threshold: 0.001,
            octaves: 4,
            octave_layers: 4,
            diffusivity: KAZE_DiffusivityType::DIFF_PM_G2,
            max_features: 1000,
        })
    }

    pub fn with_threshold(mut self, threshold: f32) -> Result<Self> {
        self.threshold = threshold;
        *self.detector.borrow_mut() = AKAZE::create(
            AKAZE_DescriptorType::DESCRIPTOR_MLDB,
            0,
            3,
            threshold,
            self.octaves,
            self.octave_layers,
            self.diffusivity,
            -1,
        )?;
        Ok(self)
    }

    pub fn with_octaves(mut self, octaves: i32) -> Result<Self> {
        self.octaves = octaves;
        *self.detector.borrow_mut() = AKAZE::create(
            AKAZE_DescriptorType::DESCRIPTOR_MLDB,
            0,
            3,
            self.threshold,
            octaves,
            self.octave_layers,
            self.diffusivity,
            -1,
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

        self.matcher
            .borrow()
            .train_match(desc1, desc2, &mut matches, &no_array())?;

        // Apply ratio test for better matches
        let mut good_matches = Vec::new();
        let mut matches_vec = matches.to_vec();

        // Sort by distance
        matches_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep matches with good distance ratio
        for m in matches_vec {
            if m.distance < 50.0 {
                // Hamming distance threshold for binary descriptors
                good_matches.push(m);
            }
        }

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

impl AlignmentAlgorithm for OpenCVAKAZE {
    fn name(&self) -> &str {
        "OpenCV-AKAZE"
    }

    fn configure(&mut self, config: &AlgorithmConfig) -> crate::Result<()> {
        if let Some(threshold) = config.parameters.get("threshold").and_then(|v| v.as_f64()) {
            *self = std::mem::take(self).with_threshold(threshold as f32)?;
        }

        if let Some(octaves) = config.parameters.get("octaves").and_then(|v| v.as_i64()) {
            *self = std::mem::take(self).with_octaves(octaves as i32)?;
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
        // AKAZE works well with the original image, minimal preprocessing needed
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
                scale: 1.0,
                skew: None,
            }),
        })
    }

    fn supports_gpu(&self) -> bool {
        false // CPU implementation
    }

    fn estimated_complexity(&self) -> ComplexityClass {
        ComplexityClass::Medium // AKAZE is faster than SIFT but slower than simple methods
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_test_pattern(width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            // Create a pattern with corners and edges for feature detection
            if (x % 16 < 8) ^ (y % 16 < 8) {
                Luma([255])
            } else {
                Luma([64])
            }
        })
    }


    #[test]
    fn test_akaze_creation() {
        let akaze = OpenCVAKAZE::new();
        assert!(akaze.is_ok());
    }

    #[test]
    fn test_akaze_with_params() {
        let akaze_result = OpenCVAKAZE::new()
            .unwrap()
            .with_threshold(0.002)
            .unwrap()
            .with_max_features(500);

        assert_eq!(akaze_result.max_features, 500);
    }

    #[test]
    fn test_akaze_alignment() {
        use crate::utils::grayimage_to_mat;
        
        let akaze = OpenCVAKAZE::new().unwrap();
        let template = create_test_pattern(64, 64);
        let target = create_test_pattern(64, 64);

        let template_mat = grayimage_to_mat(&template).unwrap();
        let target_mat = grayimage_to_mat(&target).unwrap();

        let result = akaze.align(&target_mat, &template_mat);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        assert_eq!(alignment.algorithm_name, "OpenCV-AKAZE");
        assert!(alignment.execution_time_ms >= 0.0);
        assert!(alignment.confidence >= 0.0 && alignment.confidence <= 1.0);
    }

    #[test]
    fn test_grayimage_to_mat_conversion() {
        use crate::utils::grayimage_to_mat;
        
        let image = create_test_pattern(32, 32);

        let mat_result = grayimage_to_mat(&image);
        assert!(mat_result.is_ok());

        let mat = mat_result.unwrap();
        assert_eq!(mat.cols(), 32);
        assert_eq!(mat.rows(), 32);
        assert_eq!(mat.channels(), 1);
    }
}
