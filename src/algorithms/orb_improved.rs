use crate::{AlignmentResult, algorithms::AlignmentAlgorithm};
use image::{GrayImage, imageops};
use instant::Instant;
use std::cmp::Ordering;
use rayon::prelude::*;

/// Improved ORB implementation with scale invariance and optimizations
pub struct ImprovedORB {
    pub fast_threshold: u8,
    pub max_keypoints: usize,
    pub patch_size: u32,
    pub pyramid_levels: u8,
    pub scale_factor: f32,
}

impl Default for ImprovedORB {
    fn default() -> Self {
        Self::new()
    }
}

impl ImprovedORB {
    pub fn new() -> Self {
        Self {
            fast_threshold: 20,
            max_keypoints: 500,
            patch_size: 31,
            pyramid_levels: 8,
            scale_factor: 1.2, // ORB typically uses 1.2
        }
    }
    
    pub fn with_params(threshold: u8, max_keypoints: usize, pyramid_levels: u8) -> Self {
        Self {
            fast_threshold: threshold,
            max_keypoints,
            patch_size: 31,
            pyramid_levels,
            scale_factor: 1.2,
        }
    }
}

impl AlignmentAlgorithm for ImprovedORB {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult> {
        let start = Instant::now();
        
        // Extract features with scale invariance using pyramid
        let template_features = self.extract_features_pyramid(template)?;
        let target_features = self.extract_features_pyramid(target)?;
        
        // Match features between images
        let matches = self.match_features(&template_features, &target_features);
        
        // Estimate transformation using RANSAC
        let transformation = self.estimate_transformation_ransac(&matches, &template_features, &target_features);
        
        let mut result = AlignmentResult::new("ImprovedORB");
        result.translation = transformation.translation;
        result.rotation = transformation.rotation;
        result.scale = transformation.scale;
        result.confidence = transformation.confidence;
        result.processing_time_ms = start.elapsed().as_millis() as f32;
        
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "ImprovedORB"
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImprovedKeypoint {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub angle: f32,
    pub octave: u8,     // Pyramid level
    pub scale: f32,     // Scale relative to original image
}

#[derive(Debug, Clone)]
pub struct ImprovedFeature {
    pub keypoint: ImprovedKeypoint,
    pub descriptor: [u8; 32], // 256-bit descriptor as 32 bytes
}

#[derive(Debug)]
pub struct ImprovedFeatureMatch {
    pub template_idx: usize,
    pub target_idx: usize,
    pub distance: u32,
}

#[derive(Debug)]
struct Transformation {
    translation: (f32, f32),
    rotation: f32,
    scale: f32,
    confidence: f32,
}

// Pre-computed ORB pattern (learned from training data)
// This is a simplified version - real ORB uses a specific learned pattern
const ORB_PATTERN: [(i8, i8, i8, i8); 256] = [
    (8, -3, 9, 5), (-11, 9, -8, 2), (3, -12, -13, 2), (-3, -7, -4, 5),
    (1, -11, 12, -2), (1, -1, 11, -1), (4, -2, -5, -8), (2, -13, -8, 9),
    (-11, 1, 6, 2), (11, 11, 12, -1), (6, -12, -9, -8), (12, 5, 3, -6),
    (1, 1, -4, -1), (7, -4, -6, 7), (-3, 2, 9, -8), (-4, -8, 3, 3),
    (-5, 3, 0, -4), (2, -11, -13, 0), (10, 5, 5, 2), (0, 9, 10, -3),
    (5, -8, -10, 1), (8, 3, -8, -5), (2, -6, -9, -4), (-12, 2, 0, -10),
    (5, -10, -7, -2), (-7, 9, -1, 0), (0, -1, -3, 3), (-12, 5, -2, -1),
    (-1, 1, -5, -11), (-1, 2, -3, 0), (-5, -6, 7, -1), (4, 7, 0, -8),
    (-9, 9, 3, -13), (7, -3, 13, -7), (10, -4, -5, 3), (6, 1, -13, -13),
    (-12, -11, 7, 0), (0, -1, -8, -6), (-10, -5, -6, 7), (10, 2, -6, -12),
    (-11, 8, 4, -2), (9, 0, -11, -4), (0, 11, 6, -11), (4, 1, -10, -3),
    (-6, 12, 1, 12), (-4, -8, 8, -7), (-3, 0, 8, 3), (3, 3, -3, -1),
    (-6, -11, -2, 12), (0, -3, -6, -3), (-6, 3, -12, -8), (6, 3, -2, -10),
    (-3, -10, -1, 0), (11, 2, 11, 3), (1, -8, -10, 8), (2, -2, -7, 8),
    (0, -13, 13, 0), (6, -9, -1, -1), (7, 5, 6, 3), (-13, 7, -7, -7),
    (-5, -13, 5, -11), (6, 7, -2, 12), (-6, -11, 8, 6), (-2, -2, -5, 9),
    (5, 4, 7, -6), (0, 11, -4, -5), (10, 1, 2, -8), (-3, -10, -10, -10),
    (1, 9, 6, -5), (-7, -11, 11, 3), (11, -2, -4, 3), (7, -1, 5, 12),
    (-5, 5, -2, -5), (8, -11, -1, -13), (-13, 2, -11, -8), (-2, 9, 5, 0),
    (2, -5, 2, 0), (3, -13, -12, 9), (6, -3, 5, 4), (10, 10, 1, -9),
    (-13, -8, -4, 10), (2, -2, -3, 8), (-13, -11, -8, -3), (2, -4, -7, -3),
    (12, 0, -2, 13), (-11, 7, -10, -1), (-5, -10, 0, -11), (6, 7, 12, -3),
    (-1, -1, 8, -6), (-6, 3, -1, -3), (-2, -11, -11, -3), (12, -2, 3, -10),
    (-11, -1, -2, -8), (3, -1, 7, 3), (2, -2, -12, 12), (6, -4, 12, -2),
    (-3, 11, 2, -12), (-1, 3, 2, 3), (1, 3, -11, -3), (2, -8, -7, -5),
    (0, -5, -11, -6), (-12, 8, -2, 9), (3, -7, 9, -8), (-10, -6, -1, -11),
    (11, -6, -3, -13), (3, 0, 0, -8), (-5, -2, -1, -13), (-8, -5, -10, -13),
    (7, -13, 0, -3), (1, -4, -1, -13), (6, -5, -7, 8), (8, 7, -5, -13),
    (2, 0, -8, -6), (-8, -3, -13, -6), (-6, 5, 0, 6), (-8, 8, -9, 1),
    (10, 1, -9, 4), (-4, -8, -5, 7), (7, 7, 10, -8), (-7, -3, -1, 1),
    (10, -1, 3, 1), (5, 6, -10, -8), (-6, -13, 5, -8), (4, -3, -4, -13),
    (-3, 4, -2, -13), (10, -11, 9, 11), (-9, 0, 12, 2), (-4, -2, 13, -6),
    (2, -10, -6, 1), (11, -13, 4, -13), (1, -1, 1, 9), (1, -5, -13, -5),
    (7, 4, 12, -7), (0, -2, -8, 3), (7, 2, 2, -8), (-2, 7, -12, -4),
    (1, 11, 6, -2), (-1, -1, -4, 10), (0, 8, 0, -13), (3, 12, 5, -13),
    (-9, -1, 9, -13), (12, 4, -6, -4), (-13, 13, 1, -4), (0, -2, -7, -9),
    (10, -8, -13, 3), (2, -13, 6, 8), (10, -6, -7, 0), (-11, 7, -1, -7),
    (12, 0, 5, -4), (-7, -8, 4, -12), (-13, 5, -5, -2), (0, 5, 4, 4),
    (-2, -11, -1, 8), (9, 3, -1, -12), (0, 6, -10, 12), (1, -8, -7, -10),
    (-6, 4, -6, 3), (5, 1, -3, -9), (-6, 6, -6, 3), (7, -8, 1, -7),
    (3, 8, -9, -5), (2, -4, 5, 7), (11, 4, 6, -3), (-8, -1, 11, -1),
    (-3, -6, -10, -8), (2, 7, 3, -12), (-4, -10, 12, -3), (1, -2, -4, 6),
    (3, 11, -11, 0), (-6, 2, 3, -8), (6, 12, 0, -13), (3, 2, -2, -5),
    (-4, 1, -6, 5), (-12, 0, -13, 9), (-6, 2, 7, -8), (-2, -4, -6, 5),
    (0, 0, 0, -13), (9, -13, -2, 0), (3, -13, 5, -12), (10, 11, -13, -13),
    (-2, 3, -12, 3), (11, 7, -7, 0), (12, 2, 1, -13), (12, -11, 12, -8),
    (-7, -2, -4, -7), (7, 5, -1, -13), (-5, -8, -9, 10), (6, 0, -3, -13),
    (12, 4, -13, 1), (-7, 8, 8, -3), (10, -4, 0, -13), (2, 1, -7, 0),
    (-5, 4, 2, -8), (12, 8, 4, -13), (8, 7, -10, 0), (-3, 6, -2, 4),
    (-5, -1, -8, -12), (4, -1, -2, -10), (6, -4, -13, 9), (-7, 8, -6, -12),
    (-10, 2, -13, 10), (-1, -7, 0, 2), (-5, 6, -5, -12), (6, -13, 7, -3),
    (-13, 2, -1, 8), (2, 8, -13, 0), (-6, -9, 1, -4), (-9, 13, 0, -13),
    (-2, -3, 8, 0), (4, 0, -11, 12), (0, 3, -10, 10), (-6, -9, -3, -2),
    (9, -4, -6, 2), (5, 0, -13, -10), (-3, -8, -13, 3), (-12, -1, -4, -2),
    (7, -9, -4, 3), (-8, -4, 1, 11), (11, 6, 2, -12), (6, 6, -8, 12),
    (-3, -8, 2, -10), (2, 5, -8, 8), (-9, 8, -6, -8), (-4, 0, -11, -7),
    (7, 6, -3, 8), (-5, 7, -12, 5), (2, -8, -5, 1), (0, 4, -5, -3),
    (9, -9, -6, -12), (0, -13, 0, -13), (-7, -11, -3, -13), (6, -12, -7, 10),
    (6, -8, -13, 7), (8, 7, -11, -1), (-11, -5, -6, 9), (6, 4, 2, -13),
    (-1, -6, 3, -9), (1, -4, 4, -3), (-6, 8, -12, 0), (-11, 3, -6, 2),
    (7, -10, 11, -6), (5, 0, 12, -13), (4, -8, 1, -1), (-13, 12, -6, 3),
    (1, 4, -9, -2), (-8, -12, -8, 7), (-9, 5, 0, -5), (9, 7, 5, 3),
    (-12, -2, 8, -8), (3, 7, 12, -8), (-13, 3, -1, -1), (-10, -4, -10, 12),
    (5, -2, 0, 13), (-7, 1, -12, 8), (2, 9, -5, -11), (11, -13, 0, 2)
];

impl ImprovedORB {
    /// Extract features from multiple scales using image pyramid
    fn extract_features_pyramid(&self, image: &GrayImage) -> crate::Result<Vec<ImprovedFeature>> {
        let mut all_features = Vec::new();
        let mut pyramid = self.build_pyramid(image);
        
        // Extract features from each pyramid level in parallel
        let pyramid_features: Vec<Vec<ImprovedFeature>> = pyramid
            .par_iter_mut()
            .enumerate()
            .map(|(octave, (level_image, scale))| {
                let features = self.extract_features_single_scale(level_image, octave as u8, *scale);
                features.unwrap_or_default()
            })
            .collect();
        
        // Combine features from all levels
        for level_features in pyramid_features {
            all_features.extend(level_features);
        }
        
        // Keep only the best features across all scales
        self.retain_best_features(all_features)
    }
    
    /// Build image pyramid with specified levels
    fn build_pyramid(&self, image: &GrayImage) -> Vec<(GrayImage, f32)> {
        let mut pyramid = Vec::with_capacity(self.pyramid_levels as usize);
        pyramid.push((image.clone(), 1.0));
        
        let mut current_image = image.clone();
        let mut current_scale = 1.0;
        
        for _ in 1..self.pyramid_levels {
            let new_width = (current_image.width() as f32 / self.scale_factor) as u32;
            let new_height = (current_image.height() as f32 / self.scale_factor) as u32;
            
            if new_width < 40 || new_height < 40 {
                break; // Stop if image becomes too small
            }
            
            current_scale *= self.scale_factor;
            current_image = imageops::resize(
                &current_image, 
                new_width, 
                new_height, 
                imageops::FilterType::Gaussian
            );
            
            pyramid.push((current_image.clone(), current_scale));
        }
        
        pyramid
    }
    
    /// Extract features from a single scale
    fn extract_features_single_scale(
        &self, 
        image: &GrayImage, 
        octave: u8, 
        scale: f32
    ) -> crate::Result<Vec<ImprovedFeature>> {
        // Detect FAST corners with optimization
        let corners = self.detect_fast_corners_optimized(image);
        
        // Compute orientations in parallel
        let oriented_corners: Vec<ImprovedKeypoint> = corners
            .par_iter()
            .map(|corner| {
                let mut kp = *corner;
                kp.angle = self.compute_keypoint_orientation(image, corner.x as u32, corner.y as u32);
                kp.octave = octave;
                kp.scale = scale;
                // Scale coordinates back to original image space
                kp.x *= scale;
                kp.y *= scale;
                kp
            })
            .collect();
        
        // Extract BRIEF descriptors in parallel
        let features: Vec<ImprovedFeature> = oriented_corners
            .par_iter()
            .map(|corner| {
                let descriptor = self.compute_brief_descriptor_fast(image, corner, scale);
                ImprovedFeature { 
                    keypoint: *corner, 
                    descriptor 
                }
            })
            .collect();
        
        Ok(features)
    }
    
    /// Optimized FAST corner detection with early rejection
    fn detect_fast_corners_optimized(&self, image: &GrayImage) -> Vec<ImprovedKeypoint> {
        let (width, height) = (image.width(), image.height());
        let corners: Vec<ImprovedKeypoint> = (3..(height - 3))
            .into_par_iter()
            .flat_map(|y| {
                (3..(width - 3))
                    .into_par_iter()
                    .filter_map(move |x| {
                        let center_intensity = image.get_pixel(x, y)[0];
                        
                        // Quick check: test 4 cardinal points first
                        if !self.fast_pre_check(image, x, y, center_intensity) {
                            return None;
                        }
                        
                        // Full FAST check
                        if self.is_fast_corner(image, x, y, center_intensity) {
                            let response = self.compute_corner_response(image, x, y);
                            Some(ImprovedKeypoint {
                                x: x as f32,
                                y: y as f32,
                                response,
                                angle: 0.0,
                                octave: 0,
                                scale: 1.0,
                            })
                        } else {
                            None
                        }
                    })
            })
            .collect();
        
        self.non_maximum_suppression_grid(corners)
    }
    
    /// Quick pre-check for FAST using 4 cardinal points
    fn fast_pre_check(&self, image: &GrayImage, x: u32, y: u32, center: u8) -> bool {
        let threshold = self.fast_threshold;
        let bright_threshold = center.saturating_add(threshold);
        let dark_threshold = center.saturating_sub(threshold);
        
        // Check 4 cardinal points (N, E, S, W)
        let pixels = [
            image.get_pixel(x, y - 3)[0],      // North
            image.get_pixel(x + 3, y)[0],      // East
            image.get_pixel(x, y + 3)[0],      // South
            image.get_pixel(x - 3, y)[0],      // West
        ];
        
        let bright_count = pixels.iter().filter(|&&p| p > bright_threshold).count();
        let dark_count = pixels.iter().filter(|&&p| p < dark_threshold).count();
        
        // Need at least 3 out of 4 to be all bright or all dark
        bright_count >= 3 || dark_count >= 3
    }
    
    /// Grid-based non-maximum suppression for better performance
    fn non_maximum_suppression_grid(&self, mut corners: Vec<ImprovedKeypoint>) -> Vec<ImprovedKeypoint> {
        if corners.is_empty() {
            return corners;
        }
        
        // Sort by response strength (descending)
        corners.par_sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(Ordering::Equal));
        
        let mut selected = Vec::new();
        let suppression_radius = 5.0;
        let mut suppression_map = std::collections::HashSet::new();
        
        for corner in corners {
            let grid_x = (corner.x / suppression_radius) as i32;
            let grid_y = (corner.y / suppression_radius) as i32;
            
            // Check 3x3 grid around this point
            let mut is_maximum = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if suppression_map.contains(&(grid_x + dx, grid_y + dy)) {
                        is_maximum = false;
                        break;
                    }
                }
                if !is_maximum { break; }
            }
            
            if is_maximum {
                suppression_map.insert((grid_x, grid_y));
                selected.push(corner);
                if selected.len() >= self.max_keypoints {
                    break;
                }
            }
        }
        
        selected
    }
    
    /// Optimized BRIEF descriptor using pre-computed pattern
    fn compute_brief_descriptor_fast(&self, image: &GrayImage, keypoint: &ImprovedKeypoint, scale: f32) -> [u8; 32] {
        let mut descriptor = [0u8; 32];
        let x = (keypoint.x / scale) as i32;
        let y = (keypoint.y / scale) as i32;
        let angle = keypoint.angle;
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        // Use pre-computed ORB pattern
        for (byte_idx, byte_tests) in ORB_PATTERN.chunks(8).enumerate() {
            let mut byte_val = 0u8;
            
            for (bit_idx, &(dx1, dy1, dx2, dy2)) in byte_tests.iter().enumerate() {
                // Rotate test points according to keypoint orientation
                let rx1 = (dx1 as f32 * cos_angle - dy1 as f32 * sin_angle) as i32;
                let ry1 = (dx1 as f32 * sin_angle + dy1 as f32 * cos_angle) as i32;
                let rx2 = (dx2 as f32 * cos_angle - dy2 as f32 * sin_angle) as i32;
                let ry2 = (dx2 as f32 * sin_angle + dy2 as f32 * cos_angle) as i32;
                
                let p1_x = (x + rx1).max(0).min(image.width() as i32 - 1) as u32;
                let p1_y = (y + ry1).max(0).min(image.height() as i32 - 1) as u32;
                let p2_x = (x + rx2).max(0).min(image.width() as i32 - 1) as u32;
                let p2_y = (y + ry2).max(0).min(image.height() as i32 - 1) as u32;
                
                let intensity1 = image.get_pixel(p1_x, p1_y)[0];
                let intensity2 = image.get_pixel(p2_x, p2_y)[0];
                
                if intensity1 < intensity2 {
                    byte_val |= 1 << bit_idx;
                }
            }
            
            descriptor[byte_idx] = byte_val;
        }
        
        descriptor
    }
    
    /// Keep only the best features across all scales
    fn retain_best_features(&self, mut features: Vec<ImprovedFeature>) -> crate::Result<Vec<ImprovedFeature>> {
        // Sort by keypoint response
        features.par_sort_by(|a, b| 
            b.keypoint.response.partial_cmp(&a.keypoint.response).unwrap_or(Ordering::Equal)
        );
        
        // Keep only top N features
        features.truncate(self.max_keypoints);
        Ok(features)
    }
    
    // The remaining methods (is_fast_corner, compute_corner_response, compute_keypoint_orientation)
    // remain the same as in the original FastORB implementation
    
    fn is_fast_corner(&self, image: &GrayImage, x: u32, y: u32, center: u8) -> bool {
        let offsets = [
            (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1),
            (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1),
            (-3, 0), (-3, -1), (-2, -2), (-1, -3)
        ];
        
        let threshold = self.fast_threshold;
        let bright_threshold = center.saturating_add(threshold);
        let dark_threshold = center.saturating_sub(threshold);
        
        let mut max_bright_sequence = 0;
        let mut max_dark_sequence = 0;
        let mut current_bright_sequence = 0;
        let mut current_dark_sequence = 0;
        
        // Check each pixel in the circle twice to handle wraparound
        for i in 0..(offsets.len() * 2) {
            let (dx, dy) = offsets[i % offsets.len()];
            let px = (x as i32 + dx) as u32;
            let py = (y as i32 + dy) as u32;
            let pixel = image.get_pixel(px, py)[0];
            
            if pixel > bright_threshold {
                current_bright_sequence += 1;
                current_dark_sequence = 0;
                max_bright_sequence = max_bright_sequence.max(current_bright_sequence);
            } else if pixel < dark_threshold {
                current_dark_sequence += 1;
                current_bright_sequence = 0;
                max_dark_sequence = max_dark_sequence.max(current_dark_sequence);
            } else {
                current_bright_sequence = 0;
                current_dark_sequence = 0;
            }
        }
        
        max_bright_sequence >= 9 || max_dark_sequence >= 9
    }
    
    fn compute_corner_response(&self, image: &GrayImage, x: u32, y: u32) -> f32 {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0;
        
        for dy in -2..=2 {
            for dx in -2..=2 {
                let px = (x as i32 + dx) as u32;
                let py = (y as i32 + dy) as u32;
                if px < image.width() && py < image.height() {
                    let intensity = image.get_pixel(px, py)[0] as f32;
                    sum += intensity;
                    sum_sq += intensity * intensity;
                    count += 1;
                }
            }
        }
        
        let mean = sum / count as f32;
        let variance = (sum_sq / count as f32) - (mean * mean);
        variance.sqrt()
    }
    
    fn compute_keypoint_orientation(&self, image: &GrayImage, x: u32, y: u32) -> f32 {
        let radius = 15;
        let mut m01 = 0.0;
        let mut m10 = 0.0;
        
        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                let px = x as i32 + dx;
                let py = y as i32 + dy;
                
                if px >= 0 && py >= 0 && 
                   (px as u32) < image.width() && (py as u32) < image.height() {
                    
                    let distance_sq = dx * dx + dy * dy;
                    if distance_sq <= radius * radius {
                        let intensity = image.get_pixel(px as u32, py as u32)[0] as f32;
                        m01 += intensity * dy as f32;
                        m10 += intensity * dx as f32;
                    }
                }
            }
        }
        
        m01.atan2(m10)
    }
    
    /// Match features with Hamming distance
    fn match_features(&self, template_features: &[ImprovedFeature], target_features: &[ImprovedFeature]) -> Vec<ImprovedFeatureMatch> {
        template_features
            .par_iter()
            .enumerate()
            .filter_map(|(template_idx, template_feature)| {
                let mut best_distance = u32::MAX;
                let mut second_best_distance = u32::MAX;
                let mut best_target_idx = 0;
                
                for (target_idx, target_feature) in target_features.iter().enumerate() {
                    let distance = self.hamming_distance(&template_feature.descriptor, &target_feature.descriptor);
                    
                    if distance < best_distance {
                        second_best_distance = best_distance;
                        best_distance = distance;
                        best_target_idx = target_idx;
                    } else if distance < second_best_distance {
                        second_best_distance = distance;
                    }
                }
                
                // Lowe's ratio test
                if best_distance < 80 && 
                   second_best_distance > 0 && 
                   (best_distance as f32 / second_best_distance as f32) < 0.7 {
                    Some(ImprovedFeatureMatch {
                        template_idx,
                        target_idx: best_target_idx,
                        distance: best_distance,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
    
    fn hamming_distance(&self, desc1: &[u8; 32], desc2: &[u8; 32]) -> u32 {
        desc1.iter()
            .zip(desc2.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
    
    /// RANSAC-based transformation estimation
    fn estimate_transformation_ransac(
        &self, 
        matches: &[ImprovedFeatureMatch], 
        template_features: &[ImprovedFeature], 
        target_features: &[ImprovedFeature]
    ) -> Transformation {
        if matches.len() < 3 {
            return Transformation {
                translation: (0.0, 0.0),
                rotation: 0.0,
                scale: 1.0,
                confidence: 0.0,
            };
        }
        
        let max_iterations = 100;
        let inlier_threshold = 5.0; // pixels
        let mut best_model = Transformation {
            translation: (0.0, 0.0),
            rotation: 0.0,
            scale: 1.0,
            confidence: 0.0,
        };
        let mut best_inliers = 0;
        
        // RANSAC iterations
        for _ in 0..max_iterations {
            // Randomly select 2 matches (minimum for similarity transform)
            let sample_indices = self.random_sample(matches.len(), 2);
            if sample_indices.len() < 2 {
                continue;
            }
            
            // Estimate model from sample
            let model = self.estimate_similarity_transform(
                &sample_indices,
                matches,
                template_features,
                target_features
            );
            
            // Count inliers
            let inliers = self.count_inliers(
                &model,
                matches,
                template_features,
                target_features,
                inlier_threshold
            );
            
            if inliers > best_inliers {
                best_inliers = inliers;
                best_model = model;
                best_model.confidence = inliers as f32 / matches.len() as f32;
                
                // Early termination if we have enough inliers
                if best_model.confidence > 0.8 {
                    break;
                }
            }
        }
        
        // Refine using all inliers
        if best_inliers >= 2 {
            let inlier_indices = self.get_inlier_indices(
                &best_model,
                matches,
                template_features,
                target_features,
                inlier_threshold
            );
            
            best_model = self.estimate_similarity_transform(
                &inlier_indices,
                matches,
                template_features,
                target_features
            );
            best_model.confidence = best_inliers as f32 / matches.len() as f32;
        }
        
        best_model
    }
    
    /// Random sampling for RANSAC
    fn random_sample(&self, n: usize, k: usize) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::thread_rng());
        indices.truncate(k);
        indices
    }
    
    /// Estimate similarity transform from point correspondences
    fn estimate_similarity_transform(
        &self,
        indices: &[usize],
        matches: &[ImprovedFeatureMatch],
        template_features: &[ImprovedFeature],
        target_features: &[ImprovedFeature]
    ) -> Transformation {
        if indices.len() < 2 {
            return Transformation {
                translation: (0.0, 0.0),
                rotation: 0.0,
                scale: 1.0,
                confidence: 0.0,
            };
        }
        
        // Get point pairs
        let mut src_points = Vec::new();
        let mut dst_points = Vec::new();
        
        for &idx in indices {
            let match_pair = &matches[idx];
            let src = &template_features[match_pair.template_idx].keypoint;
            let dst = &target_features[match_pair.target_idx].keypoint;
            src_points.push((src.x, src.y));
            dst_points.push((dst.x, dst.y));
        }
        
        // Estimate transform using least squares
        // For simplicity, we'll use a basic approach here
        // In production, you'd use a proper similarity transform solver
        
        // Calculate centroids
        let src_centroid = (
            src_points.iter().map(|p| p.0).sum::<f32>() / src_points.len() as f32,
            src_points.iter().map(|p| p.1).sum::<f32>() / src_points.len() as f32,
        );
        let dst_centroid = (
            dst_points.iter().map(|p| p.0).sum::<f32>() / dst_points.len() as f32,
            dst_points.iter().map(|p| p.1).sum::<f32>() / dst_points.len() as f32,
        );
        
        // Estimate scale and rotation
        let mut scale_sum = 0.0;
        let mut angle_sum = 0.0;
        let mut count = 0;
        
        for i in 0..src_points.len() {
            for j in i+1..src_points.len() {
                // Source vector
                let src_dx = src_points[j].0 - src_points[i].0;
                let src_dy = src_points[j].1 - src_points[i].1;
                let src_dist = (src_dx * src_dx + src_dy * src_dy).sqrt();
                
                // Destination vector
                let dst_dx = dst_points[j].0 - dst_points[i].0;
                let dst_dy = dst_points[j].1 - dst_points[i].1;
                let dst_dist = (dst_dx * dst_dx + dst_dy * dst_dy).sqrt();
                
                if src_dist > 0.0 && dst_dist > 0.0 {
                    scale_sum += dst_dist / src_dist;
                    
                    let src_angle = src_dy.atan2(src_dx);
                    let dst_angle = dst_dy.atan2(dst_dx);
                    let mut angle_diff = dst_angle - src_angle;
                    
                    // Normalize angle to [-π, π]
                    while angle_diff > std::f32::consts::PI {
                        angle_diff -= 2.0 * std::f32::consts::PI;
                    }
                    while angle_diff < -std::f32::consts::PI {
                        angle_diff += 2.0 * std::f32::consts::PI;
                    }
                    
                    angle_sum += angle_diff;
                    count += 1;
                }
            }
        }
        
        let scale = if count > 0 { scale_sum / count as f32 } else { 1.0 };
        let rotation = if count > 0 { angle_sum / count as f32 } else { 0.0 };
        
        // Calculate translation after applying scale and rotation
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let translation = (
            dst_centroid.0 - scale * (src_centroid.0 * cos_r - src_centroid.1 * sin_r),
            dst_centroid.1 - scale * (src_centroid.0 * sin_r + src_centroid.1 * cos_r),
        );
        
        Transformation {
            translation,
            rotation: rotation * 180.0 / std::f32::consts::PI,
            scale,
            confidence: 0.0,
        }
    }
    
    /// Count inliers for a given transformation model
    fn count_inliers(
        &self,
        model: &Transformation,
        matches: &[ImprovedFeatureMatch],
        template_features: &[ImprovedFeature],
        target_features: &[ImprovedFeature],
        threshold: f32
    ) -> usize {
        matches.iter().filter(|match_pair| {
            let src = &template_features[match_pair.template_idx].keypoint;
            let dst = &target_features[match_pair.target_idx].keypoint;
            
            // Apply transformation to source point
            let angle_rad = model.rotation * std::f32::consts::PI / 180.0;
            let cos_r = angle_rad.cos();
            let sin_r = angle_rad.sin();
            
            let transformed_x = model.scale * (src.x * cos_r - src.y * sin_r) + model.translation.0;
            let transformed_y = model.scale * (src.x * sin_r + src.y * cos_r) + model.translation.1;
            
            // Calculate error
            let dx = transformed_x - dst.x;
            let dy = transformed_y - dst.y;
            let error = (dx * dx + dy * dy).sqrt();
            
            error < threshold
        }).count()
    }
    
    /// Get indices of inliers for refinement
    fn get_inlier_indices(
        &self,
        model: &Transformation,
        matches: &[ImprovedFeatureMatch],
        template_features: &[ImprovedFeature],
        target_features: &[ImprovedFeature],
        threshold: f32
    ) -> Vec<usize> {
        matches.iter()
            .enumerate()
            .filter_map(|(idx, match_pair)| {
                let src = &template_features[match_pair.template_idx].keypoint;
                let dst = &target_features[match_pair.target_idx].keypoint;
                
                // Apply transformation
                let angle_rad = model.rotation * std::f32::consts::PI / 180.0;
                let cos_r = angle_rad.cos();
                let sin_r = angle_rad.sin();
                
                let transformed_x = model.scale * (src.x * cos_r - src.y * sin_r) + model.translation.0;
                let transformed_y = model.scale * (src.x * sin_r + src.y * cos_r) + model.translation.1;
                
                let dx = transformed_x - dst.x;
                let dy = transformed_y - dst.y;
                let error = (dx * dx + dy * dy).sqrt();
                
                if error < threshold {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }
}