use crate::pipeline::{AlgorithmConfig, AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use crate::Result;
use opencv::core::{no_array, Mat, Point2i, Scalar, Size, TermCriteria};
use opencv::imgproc;
use opencv::prelude::*;
use opencv::video;
use std::collections::HashMap;
use std::time::Instant;

/// Enhanced Correlation Coefficient (ECC) image alignment
/// ECC is excellent for handling complex illumination changes and geometric distortions
pub struct OpenCVECC {
    motion_type: MotionType,
    max_iterations: i32,
    termination_eps: f64,
    gaussian_filter_size: i32,
}

#[derive(Debug, Clone, Copy)]
pub enum MotionType {
    Translation,
    Euclidean,
    Affine,
    Homography,
}

impl Default for OpenCVECC {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenCVECC {
    pub fn new() -> Self {
        Self {
            motion_type: MotionType::Euclidean,
            max_iterations: 50,
            termination_eps: 1e-10,
            gaussian_filter_size: 5,
        }
    }

    pub fn with_motion_type(mut self, motion_type: MotionType) -> Self {
        self.motion_type = motion_type;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: i32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_termination_eps(mut self, eps: f64) -> Self {
        self.termination_eps = eps;
        self
    }

    pub fn with_gaussian_filter_size(mut self, size: i32) -> Self {
        self.gaussian_filter_size = size;
        self
    }

    fn motion_type_to_opencv(&self) -> i32 {
        match self.motion_type {
            MotionType::Translation => video::MOTION_TRANSLATION,
            MotionType::Euclidean => video::MOTION_EUCLIDEAN,
            MotionType::Affine => video::MOTION_AFFINE,
            MotionType::Homography => video::MOTION_HOMOGRAPHY,
        }
    }

    fn get_initial_warp_matrix(&self) -> Result<Mat> {
        match self.motion_type {
            MotionType::Translation => {
                // 2x3 matrix for translation
                let mut warp = Mat::eye(2, 3, opencv::core::CV_32F)?.to_mat()?;
                *warp.at_2d_mut::<f32>(0, 2)? = 0.0; // tx
                *warp.at_2d_mut::<f32>(1, 2)? = 0.0; // ty
                Ok(warp)
            }
            MotionType::Euclidean | MotionType::Affine => {
                // 2x3 matrix for Euclidean/Affine
                Ok(Mat::eye(2, 3, opencv::core::CV_32F)?.to_mat()?)
            }
            MotionType::Homography => {
                // 3x3 matrix for homography
                Ok(Mat::eye(3, 3, opencv::core::CV_32F)?.to_mat()?)
            }
        }
    }

    fn preprocess_image(&self, image: &Mat) -> Result<Mat> {
        let mut preprocessed = Mat::default();

        // Convert to float
        image.convert_to(&mut preprocessed, opencv::core::CV_32F, 1.0, 0.0)?;

        // Apply Gaussian smoothing if specified
        if self.gaussian_filter_size > 0 {
            let mut smoothed = Mat::default();
            imgproc::gaussian_blur(
                &preprocessed,
                &mut smoothed,
                Size::new(self.gaussian_filter_size, self.gaussian_filter_size),
                0.0,
                0.0,
                opencv::core::BORDER_REFLECT_101,
                opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;
            preprocessed = smoothed;
        }

        Ok(preprocessed)
    }

    fn extract_transformation_params(&self, warp_matrix: &Mat) -> Result<(f32, f32, f32, f32)> {
        match self.motion_type {
            MotionType::Translation => {
                let tx = *warp_matrix.at_2d::<f32>(0, 2)?;
                let ty = *warp_matrix.at_2d::<f32>(1, 2)?;
                Ok((tx, ty, 0.0, 1.0)) // translation only
            }
            MotionType::Euclidean => {
                let a = *warp_matrix.at_2d::<f32>(0, 0)?;
                let b = *warp_matrix.at_2d::<f32>(0, 1)?;
                let tx = *warp_matrix.at_2d::<f32>(0, 2)?;
                let ty = *warp_matrix.at_2d::<f32>(1, 2)?;

                let rotation = b.atan2(a).to_degrees();
                let scale = (a * a + b * b).sqrt();

                Ok((tx, ty, rotation, scale))
            }
            MotionType::Affine => {
                let tx = *warp_matrix.at_2d::<f32>(0, 2)?;
                let ty = *warp_matrix.at_2d::<f32>(1, 2)?;

                // Extract rotation and scale from affine matrix
                let a = *warp_matrix.at_2d::<f32>(0, 0)?;
                let b = *warp_matrix.at_2d::<f32>(0, 1)?;
                let c = *warp_matrix.at_2d::<f32>(1, 0)?;
                let d = *warp_matrix.at_2d::<f32>(1, 1)?;

                let scale_x = (a * a + c * c).sqrt();
                let scale_y = (b * b + d * d).sqrt();
                let scale = (scale_x + scale_y) / 2.0;

                let rotation = b.atan2(a).to_degrees();

                Ok((tx, ty, rotation, scale))
            }
            MotionType::Homography => {
                // For homography, extract approximate translation from the homography matrix
                let h02 = *warp_matrix.at_2d::<f32>(0, 2)?;
                let h12 = *warp_matrix.at_2d::<f32>(1, 2)?;
                let h22 = *warp_matrix.at_2d::<f32>(2, 2)?;

                // Normalize by h22 to get translation
                let tx = if h22.abs() > 1e-6 { h02 / h22 } else { 0.0 };
                let ty = if h22.abs() > 1e-6 { h12 / h22 } else { 0.0 };

                // Extract rotation and scale (approximate)
                let h00 = *warp_matrix.at_2d::<f32>(0, 0)?;
                let h01 = *warp_matrix.at_2d::<f32>(0, 1)?;
                let h10 = *warp_matrix.at_2d::<f32>(1, 0)?;
                let h11 = *warp_matrix.at_2d::<f32>(1, 1)?;

                let scale_x = (h00 * h00 + h10 * h10).sqrt();
                let scale_y = (h01 * h01 + h11 * h11).sqrt();
                let scale = (scale_x + scale_y) / 2.0;

                let rotation = h01.atan2(h00).to_degrees();

                Ok((tx, ty, rotation, scale))
            }
        }
    }

    fn calculate_correlation_score(
        &self,
        template: &Mat,
        target: &Mat,
        warp_matrix: &Mat,
    ) -> Result<f64> {
        // Warp the target image using the estimated transformation
        let mut warped = Mat::default();

        match self.motion_type {
            MotionType::Homography => {
                imgproc::warp_perspective(
                    target,
                    &mut warped,
                    warp_matrix,
                    template.size()?,
                    imgproc::INTER_LINEAR,
                    opencv::core::BORDER_CONSTANT,
                    Scalar::all(0.0),
                )?;
            }
            _ => {
                imgproc::warp_affine(
                    target,
                    &mut warped,
                    warp_matrix,
                    template.size()?,
                    imgproc::INTER_LINEAR,
                    opencv::core::BORDER_CONSTANT,
                    Scalar::all(0.0),
                )?;
            }
        }

        // Calculate normalized cross correlation
        let mut correlation = Mat::default();
        imgproc::match_template(
            &warped,
            template,
            &mut correlation,
            imgproc::TM_CCOEFF_NORMED,
            &no_array(),
        )?;

        // Get the correlation value
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = Point2i::default();
        let mut max_loc = Point2i::default();

        opencv::core::min_max_loc(
            &correlation,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &no_array(),
        )?;

        Ok(max_val.clamp(0.0, 1.0))
    }
}

impl AlignmentAlgorithm for OpenCVECC {
    fn name(&self) -> &str {
        match self.motion_type {
            MotionType::Translation => "OpenCV-ECC-Translation",
            MotionType::Euclidean => "OpenCV-ECC-Euclidean",
            MotionType::Affine => "OpenCV-ECC-Affine",
            MotionType::Homography => "OpenCV-ECC-Homography",
        }
    }

    fn configure(&mut self, config: &AlgorithmConfig) -> crate::Result<()> {
        if let Some(motion_type_str) = config
            .parameters
            .get("motion_type")
            .and_then(|v| v.as_str())
        {
            self.motion_type = match motion_type_str {
                "translation" => MotionType::Translation,
                "euclidean" => MotionType::Euclidean,
                "affine" => MotionType::Affine,
                "homography" => MotionType::Homography,
                _ => self.motion_type, // Keep current if unknown
            };
        }

        if let Some(max_iter) = config
            .parameters
            .get("max_iterations")
            .and_then(|v| v.as_i64())
        {
            self.max_iterations = max_iter as i32;
        }

        if let Some(eps) = config
            .parameters
            .get("termination_eps")
            .and_then(|v| v.as_f64())
        {
            self.termination_eps = eps;
        }

        if let Some(gauss_size) = config
            .parameters
            .get("gaussian_filter_size")
            .and_then(|v| v.as_i64())
        {
            self.gaussian_filter_size = gauss_size as i32;
        }

        Ok(())
    }

    fn preprocess(&self, image: &Mat) -> crate::Result<Mat> {
        self.preprocess_image(image)
    }

    fn align(&self, search_image: &Mat, patch: &Mat) -> crate::Result<AlignmentResult> {
        let start = Instant::now();

        // Preprocess images
        let search_processed = self.preprocess_image(search_image)?;
        let patch_processed = self.preprocess_image(patch)?;

        // Initialize warp matrix
        let mut warp_matrix = self.get_initial_warp_matrix()?;

        // Set up termination criteria
        let criteria = TermCriteria::new(
            opencv::core::TermCriteria_Type::COUNT as i32
                + opencv::core::TermCriteria_Type::EPS as i32,
            self.max_iterations,
            self.termination_eps,
        )?;

        // Perform ECC alignment
        let ecc_result = video::find_transform_ecc(
            &patch_processed,
            &search_processed,
            &mut warp_matrix,
            self.motion_type_to_opencv(),
            criteria,
            &no_array(),
            self.gaussian_filter_size,
        );

        if ecc_result.is_err() {
            // ECC failed
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

        let (tx, ty, rotation, scale) = self.extract_transformation_params(&warp_matrix)?;

        // Calculate match location in search image
        let match_x = (search_image.cols() / 2) + tx as i32;
        let match_y = (search_image.rows() / 2) + ty as i32;

        // Calculate confidence based on correlation score
        let confidence = self
            .calculate_correlation_score(&patch_processed, &search_processed, &warp_matrix)
            .unwrap_or(0.0);

        let mut metadata = HashMap::new();
        metadata.insert(
            "motion_type".to_string(),
            serde_json::Value::from(format!("{:?}", self.motion_type)),
        );
        metadata.insert(
            "max_iterations".to_string(),
            serde_json::Value::from(self.max_iterations),
        );
        metadata.insert(
            "termination_eps".to_string(),
            serde_json::Value::from(self.termination_eps),
        );

        Ok(AlignmentResult {
            location: crate::pipeline::SerializableRect {
                x: match_x.max(0),
                y: match_y.max(0),
                width: patch.cols(),
                height: patch.rows(),
            },
            score: confidence,
            confidence,
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
        match self.motion_type {
            MotionType::Translation => ComplexityClass::Low,
            MotionType::Euclidean => ComplexityClass::Medium,
            MotionType::Affine | MotionType::Homography => ComplexityClass::High,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::grayimage_to_mat;
    use image::{GrayImage, Luma};

    fn create_test_pattern(width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            // Create a gradient pattern for ECC testing
            let intensity = ((x + y) * 255 / (width + height)) as u8;
            Luma([intensity])
        })
    }


    #[test]
    fn test_ecc_creation() {
        let ecc = OpenCVECC::new();
        assert_eq!(ecc.max_iterations, 50);
        assert_eq!(ecc.termination_eps, 1e-10);
    }

    #[test]
    fn test_ecc_with_params() {
        let ecc = OpenCVECC::new()
            .with_motion_type(MotionType::Affine)
            .with_max_iterations(100)
            .with_termination_eps(1e-8);

        assert_eq!(ecc.max_iterations, 100);
        assert_eq!(ecc.termination_eps, 1e-8);
    }

    #[test]
    fn test_ecc_alignment() {
        let ecc = OpenCVECC::new().with_motion_type(MotionType::Translation);
        let template = create_test_pattern(64, 64);
        let target = create_test_pattern(64, 64);

        let template_mat = grayimage_to_mat(&template).unwrap();
        let target_mat = grayimage_to_mat(&target).unwrap();

        let result = ecc.align(&target_mat, &template_mat);
        assert!(result.is_ok());

        let alignment = result.unwrap();
        assert!(alignment.algorithm_name.contains("OpenCV-ECC"));
        assert!(alignment.execution_time_ms >= 0.0);
        assert!(alignment.confidence >= 0.0 && alignment.confidence <= 1.0);
    }

    #[test]
    fn test_motion_type_names() {
        let ecc_translation = OpenCVECC::new().with_motion_type(MotionType::Translation);
        let ecc_euclidean = OpenCVECC::new().with_motion_type(MotionType::Euclidean);
        let ecc_affine = OpenCVECC::new().with_motion_type(MotionType::Affine);
        let ecc_homography = OpenCVECC::new().with_motion_type(MotionType::Homography);

        assert_eq!(ecc_translation.name(), "OpenCV-ECC-Translation");
        assert_eq!(ecc_euclidean.name(), "OpenCV-ECC-Euclidean");
        assert_eq!(ecc_affine.name(), "OpenCV-ECC-Affine");
        assert_eq!(ecc_homography.name(), "OpenCV-ECC-Homography");
    }

    #[test]
    fn test_grayimage_to_mat_conversion() {
        let image = create_test_pattern(32, 32);

        let mat_result = grayimage_to_mat(&image);
        assert!(mat_result.is_ok());

        let mat = mat_result.unwrap();
        assert_eq!(mat.cols(), 32);
        assert_eq!(mat.rows(), 32);
        assert_eq!(mat.channels(), 1);
    }

    #[test]
    fn test_initial_warp_matrix() {
        let ecc_translation = OpenCVECC::new().with_motion_type(MotionType::Translation);
        let ecc_homography = OpenCVECC::new().with_motion_type(MotionType::Homography);

        let warp_translation = ecc_translation.get_initial_warp_matrix();
        let warp_homography = ecc_homography.get_initial_warp_matrix();

        assert!(warp_translation.is_ok());
        assert!(warp_homography.is_ok());

        let warp_t = warp_translation.unwrap();
        let warp_h = warp_homography.unwrap();

        assert_eq!(warp_t.rows(), 2);
        assert_eq!(warp_t.cols(), 3);
        assert_eq!(warp_h.rows(), 3);
        assert_eq!(warp_h.cols(), 3);
    }
}
