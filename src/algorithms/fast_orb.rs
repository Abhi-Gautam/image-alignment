use crate::{AlignmentResult, algorithms::AlignmentAlgorithm};
use image::GrayImage;
use instant::Instant;
use std::cmp::Ordering;

/// Proper ORB implementation with FAST corner detection and BRIEF descriptors
pub struct FastORB {
    pub fast_threshold: u8,
    pub max_keypoints: usize,
    pub patch_size: u32,
    pub pyramid_levels: u8,
}

impl Default for FastORB {
    fn default() -> Self {
        Self::new()
    }
}

impl FastORB {
    pub fn new() -> Self {
        Self {
            fast_threshold: 10,  // Reduced from 20 to detect more corners
            max_keypoints: 500,
            patch_size: 31,
            pyramid_levels: 3,
        }
    }
    
    pub fn with_params(threshold: u8, max_keypoints: usize) -> Self {
        Self {
            fast_threshold: threshold,
            max_keypoints,
            patch_size: 31,
            pyramid_levels: 3,
        }
    }
}

impl AlignmentAlgorithm for FastORB {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult> {
        let start = Instant::now();
        
        // Extract keypoints and descriptors from both images
        let template_features = self.extract_features(template)?;
        let target_features = self.extract_features(target)?;
        
        // Match features between images
        let matches = self.match_features(&template_features, &target_features);
        
        // Estimate transformation from matches
        let transformation = self.estimate_transformation(&matches, &template_features, &target_features);
        
        let mut result = AlignmentResult::new("FastORB");
        result.translation = transformation.translation;
        result.rotation = transformation.rotation;
        result.scale = transformation.scale;
        result.confidence = transformation.confidence;
        result.processing_time_ms = start.elapsed().as_millis() as f32;
        
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "FastORB"
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub response: f32,
    pub angle: f32,
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub keypoint: Keypoint,
    pub descriptor: [u8; 32], // 256-bit descriptor as 32 bytes
}

#[derive(Debug)]
pub struct FeatureMatch {
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

impl FastORB {
    fn extract_features(&self, image: &GrayImage) -> crate::Result<Vec<Feature>> {
        // Step 1: Detect FAST corners
        let corners = self.detect_fast_corners(image);
        
        // Step 2: Compute orientation for each corner
        let oriented_corners = self.compute_orientations(image, corners);
        
        // Step 3: Extract BRIEF descriptors
        let features = self.extract_brief_descriptors(image, oriented_corners);
        
        Ok(features)
    }
    
    fn detect_fast_corners(&self, image: &GrayImage) -> Vec<Keypoint> {
        let (width, height) = (image.width(), image.height());
        let mut corners = Vec::new();
        
        // FAST-9 detection (check 16 pixels in a circle)
        for y in 3..(height - 3) {
            for x in 3..(width - 3) {
                let center_intensity = image.get_pixel(x, y)[0];
                
                if self.is_fast_corner(image, x, y, center_intensity) {
                    let response = self.compute_corner_response(image, x, y);
                    corners.push(Keypoint {
                        x: x as f32,
                        y: y as f32,
                        response,
                        angle: 0.0, // Will be computed later
                    });
                }
            }
        }
        
        // Non-maximum suppression
        self.non_maximum_suppression(corners)
    }
    
    fn is_fast_corner(&self, image: &GrayImage, x: u32, y: u32, center: u8) -> bool {
        // FAST-9: Check if 9 contiguous pixels are all brighter or all darker
        let offsets = [
            (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1),
            (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1),
            (-3, 0), (-3, -1), (-2, -2), (-1, -3)
        ];
        
        let threshold = self.fast_threshold;
        let bright_threshold = center.saturating_add(threshold);
        let dark_threshold = center.saturating_sub(threshold);
        
        let mut _bright_count = 0;
        let mut _dark_count = 0;
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
                _bright_count += 1;
                current_bright_sequence += 1;
                current_dark_sequence = 0;
                max_bright_sequence = max_bright_sequence.max(current_bright_sequence);
            } else if pixel < dark_threshold {
                _dark_count += 1;
                current_dark_sequence += 1;
                current_bright_sequence = 0;
                max_dark_sequence = max_dark_sequence.max(current_dark_sequence);
            } else {
                current_bright_sequence = 0;
                current_dark_sequence = 0;
            }
        }
        
        // Need at least 9 contiguous pixels that are significantly different
        max_bright_sequence >= 9 || max_dark_sequence >= 9
    }
    
    fn compute_corner_response(&self, image: &GrayImage, x: u32, y: u32) -> f32 {
        // Simple corner response based on intensity variance in neighborhood
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
    
    fn non_maximum_suppression(&self, mut corners: Vec<Keypoint>) -> Vec<Keypoint> {
        if corners.is_empty() {
            return corners;
        }
        
        // Sort by response strength (descending)
        corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(Ordering::Equal));
        
        let mut selected: Vec<Keypoint> = Vec::new();
        let suppression_radius = 3.0;  // Reduced from 5.0 for less aggressive suppression
        
        for corner in corners {
            let mut is_maximum = true;
            
            // Check if this corner is too close to an already selected corner
            for selected_corner in &selected {
                let dx = corner.x - selected_corner.x;
                let dy = corner.y - selected_corner.y;
                let distance = (dx * dx + dy * dy).sqrt();
                
                if distance < suppression_radius {
                    is_maximum = false;
                    break;
                }
            }
            
            if is_maximum {
                selected.push(corner);
                if selected.len() >= self.max_keypoints {
                    break;
                }
            }
        }
        
        selected
    }
    
    fn compute_orientations(&self, image: &GrayImage, mut corners: Vec<Keypoint>) -> Vec<Keypoint> {
        for corner in &mut corners {
            corner.angle = self.compute_keypoint_orientation(image, corner.x as u32, corner.y as u32);
        }
        corners
    }
    
    fn compute_keypoint_orientation(&self, image: &GrayImage, x: u32, y: u32) -> f32 {
        // Compute orientation using intensity centroid method
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
    
    fn extract_brief_descriptors(&self, image: &GrayImage, corners: Vec<Keypoint>) -> Vec<Feature> {
        corners.into_iter()
            .map(|corner| {
                let descriptor = self.compute_brief_descriptor(image, &corner);
                Feature { keypoint: corner, descriptor }
            })
            .collect()
    }
    
    fn compute_brief_descriptor(&self, image: &GrayImage, keypoint: &Keypoint) -> [u8; 32] {
        let mut descriptor = [0u8; 32];
        let x = keypoint.x as i32;
        let y = keypoint.y as i32;
        let angle = keypoint.angle;
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        // BRIEF test pattern (256 bit pairs)
        let test_pattern = self.get_brief_pattern();
        
        for (byte_idx, byte_tests) in test_pattern.chunks(8).enumerate() {
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
    
    fn get_brief_pattern(&self) -> Vec<(i8, i8, i8, i8)> {
        // Simplified BRIEF pattern (256 test pairs)
        // In a real implementation, this would be a pre-computed optimal pattern
        let mut pattern = Vec::with_capacity(256);
        
        for i in 0..256 {
            // Generate pseudo-random test pairs within a 31x31 patch
            let seed = i as u32;
            let x1 = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) % 31) as i8 - 15;
            let y1 = (((seed + 1).wrapping_mul(1103515245).wrapping_add(12345)) % 31) as i8 - 15;
            let x2 = (((seed + 2).wrapping_mul(1103515245).wrapping_add(12345)) % 31) as i8 - 15;
            let y2 = (((seed + 3).wrapping_mul(1103515245).wrapping_add(12345)) % 31) as i8 - 15;
            
            pattern.push((x1, y1, x2, y2));
        }
        
        pattern
    }
    
    fn match_features(&self, template_features: &[Feature], target_features: &[Feature]) -> Vec<FeatureMatch> {
        let mut matches = Vec::new();
        
        for (template_idx, template_feature) in template_features.iter().enumerate() {
            let mut best_distance = u32::MAX;
            let mut second_best_distance = u32::MAX;
            let mut best_target_idx = 0;
            
            // Find best and second-best matches using Hamming distance
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
            
            // Lowe's ratio test to filter good matches - relaxed thresholds
            if best_distance < 120 && // Increased from 80 to allow more matches
               second_best_distance > 0 && 
               (best_distance as f32 / second_best_distance as f32) < 0.8 { // Increased from 0.7
                matches.push(FeatureMatch {
                    template_idx,
                    target_idx: best_target_idx,
                    distance: best_distance,
                });
            }
        }
        
        matches
    }
    
    fn hamming_distance(&self, desc1: &[u8; 32], desc2: &[u8; 32]) -> u32 {
        desc1.iter()
            .zip(desc2.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
    
    fn estimate_transformation(&self, matches: &[FeatureMatch], template_features: &[Feature], target_features: &[Feature]) -> Transformation {
        if matches.is_empty() {
            return Transformation {
                translation: (0.0, 0.0),
                rotation: 0.0,
                scale: 1.0,
                confidence: 0.0,
            };
        }
        
        // If we have 1-2 matches, provide basic translation with low confidence
        if matches.len() < 3 {
            let template_kp = &template_features[matches[0].template_idx].keypoint;
            let target_kp = &target_features[matches[0].target_idx].keypoint;
            
            return Transformation {
                translation: (target_kp.x - template_kp.x, target_kp.y - template_kp.y),
                rotation: 0.0,
                scale: 1.0,
                confidence: 0.3,  // Low but non-zero confidence
            };
        }
        
        // Simple transformation estimation using the centroid of matches
        let mut template_centroid = (0.0, 0.0);
        let mut target_centroid = (0.0, 0.0);
        
        for match_pair in matches {
            let template_kp = &template_features[match_pair.template_idx].keypoint;
            let target_kp = &target_features[match_pair.target_idx].keypoint;
            
            template_centroid.0 += template_kp.x;
            template_centroid.1 += template_kp.y;
            target_centroid.0 += target_kp.x;
            target_centroid.1 += target_kp.y;
        }
        
        let n = matches.len() as f32;
        template_centroid.0 /= n;
        template_centroid.1 /= n;
        target_centroid.0 /= n;
        target_centroid.1 /= n;
        
        // Translation is the difference between centroids
        let translation = (
            target_centroid.0 - template_centroid.0,
            target_centroid.1 - template_centroid.1
        );
        
        // Estimate rotation using matched feature orientations
        let mut angle_diffs = Vec::new();
        for match_pair in matches {
            let template_angle = template_features[match_pair.template_idx].keypoint.angle;
            let target_angle = target_features[match_pair.target_idx].keypoint.angle;
            let mut diff = target_angle - template_angle;
            
            // Normalize angle difference to [-π, π]
            while diff > std::f32::consts::PI {
                diff -= 2.0 * std::f32::consts::PI;
            }
            while diff < -std::f32::consts::PI {
                diff += 2.0 * std::f32::consts::PI;
            }
            
            angle_diffs.push(diff);
        }
        
        // Median angle difference for robust rotation estimation
        angle_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rotation = if !angle_diffs.is_empty() {
            angle_diffs[angle_diffs.len() / 2] * 180.0 / std::f32::consts::PI
        } else {
            0.0
        };
        
        // Confidence based on number of matches and consistency
        let confidence = ((matches.len() as f32 / 20.0).min(1.0) * 0.5 + 0.5).min(1.0);
        
        Transformation {
            translation,
            rotation,
            scale: 1.0, // Scale estimation would require more sophisticated analysis
            confidence,
        }
    }
}