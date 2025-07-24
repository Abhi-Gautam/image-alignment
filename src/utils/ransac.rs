use crate::Result;
use opencv::core::{DMatch, KeyPoint};
use opencv::prelude::*;
use rand::seq::SliceRandom;

/// Configuration for RANSAC transformation estimation
#[derive(Clone, Debug)]
pub struct RansacConfig {
    pub max_iterations: i32,
    pub inlier_threshold: f32,
    pub min_inliers: usize,
    pub confidence_threshold: f32,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            inlier_threshold: 3.0, // pixels
            min_inliers: 4,
            confidence_threshold: 0.99,
        }
    }
}

/// Result of RANSAC estimation
#[derive(Clone, Debug)]
pub struct RansacResult {
    pub translation: (f32, f32),
    pub rotation: f32,
    pub scale: f32,
    pub confidence: f32,
    pub inlier_count: usize,
    pub total_matches: usize,
}

/// Estimate transformation between two sets of keypoints using RANSAC
pub fn estimate_transformation_ransac(
    kp1: &[KeyPoint],
    kp2: &[KeyPoint],
    matches: &[DMatch],
    config: Option<RansacConfig>,
) -> Result<RansacResult> {
    let config = config.unwrap_or_default();
    
    if matches.len() < config.min_inliers {
        return handle_few_matches(kp1, kp2, matches);
    }

    let mut best_inlier_count = 0;
    let mut best_transformation = (0.0f32, 0.0f32, 0.0f32, 1.0f32); // tx, ty, rotation, scale
    let mut rng = rand::thread_rng();

    for _ in 0..config.max_iterations {
        if matches.len() < 4 {
            break;
        }

        // Randomly sample 4 matches for minimal set
        let indices: Vec<usize> = (0..matches.len()).collect();
        let sample_indices: Vec<usize> = indices
            .choose_multiple(&mut rng, 4.min(matches.len()))
            .cloned()
            .collect();

        // Estimate transformation from sample
        let sample_matches: Vec<&DMatch> = sample_indices
            .iter()
            .map(|&i| &matches[i])
            .collect();

        if let Ok(transformation) = estimate_from_sample(kp1, kp2, &sample_matches) {
            // Count inliers for this transformation
            let inlier_count = count_inliers(
                kp1,
                kp2,
                matches,
                &transformation,
                config.inlier_threshold,
            );

            if inlier_count > best_inlier_count {
                best_inlier_count = inlier_count;
                best_transformation = transformation;

                // Early termination if we have enough inliers
                let inlier_ratio = inlier_count as f32 / matches.len() as f32;
                if inlier_ratio > 0.8 {
                    break;
                }
            }
        }
    }

    // Calculate confidence based on inlier ratio
    let inlier_ratio = best_inlier_count as f32 / matches.len() as f32;
    let confidence = if best_inlier_count >= config.min_inliers {
        (inlier_ratio * 0.8 + 0.2).min(1.0) // Scale to [0.2, 1.0]
    } else {
        0.1 // Very low confidence for insufficient inliers
    };

    Ok(RansacResult {
        translation: (best_transformation.0, best_transformation.1),
        rotation: best_transformation.2,
        scale: best_transformation.3,
        confidence,
        inlier_count: best_inlier_count,
        total_matches: matches.len(),
    })
}

/// Handle case with few matches using simple average
fn handle_few_matches(
    kp1: &[KeyPoint],
    kp2: &[KeyPoint],
    matches: &[DMatch],
) -> Result<RansacResult> {
    if matches.is_empty() {
        return Ok(RansacResult {
            translation: (0.0, 0.0),
            rotation: 0.0,
            scale: 1.0,
            confidence: 0.0,
            inlier_count: 0,
            total_matches: 0,
        });
    }

    // Simple average for few matches
    let mut dx_sum = 0.0;
    let mut dy_sum = 0.0;
    let mut angle_sum = 0.0;
    let mut scale_sum = 0.0;
    let mut valid_scales = 0;

    for m in matches {
        let pt1 = kp1[m.query_idx as usize].pt();
        let pt2 = kp2[m.train_idx as usize].pt();

        dx_sum += pt2.x - pt1.x;
        dy_sum += pt2.y - pt1.y;

        let angle1 = kp1[m.query_idx as usize].angle();
        let angle2 = kp2[m.train_idx as usize].angle();
        angle_sum += angle2 - angle1;

        // Calculate scale from keypoint sizes if available
        let size1 = kp1[m.query_idx as usize].size();
        let size2 = kp2[m.train_idx as usize].size();
        if size1 > 0.0 && size2 > 0.0 {
            scale_sum += size2 / size1;
            valid_scales += 1;
        }
    }

    let tx = dx_sum / matches.len() as f32;
    let ty = dy_sum / matches.len() as f32;
    let rotation = angle_sum / matches.len() as f32;
    let scale = if valid_scales > 0 {
        scale_sum / valid_scales as f32
    } else {
        1.0
    };
    
    let confidence = 0.3; // Low confidence for few matches

    Ok(RansacResult {
        translation: (tx, ty),
        rotation,
        scale,
        confidence,
        inlier_count: matches.len(),
        total_matches: matches.len(),
    })
}

/// Estimate transformation from a sample of 4 matches
fn estimate_from_sample(
    kp1: &[KeyPoint],
    kp2: &[KeyPoint],
    sample_matches: &[&DMatch],
) -> Result<(f32, f32, f32, f32)> {
    if sample_matches.len() < 2 {
        return Err(anyhow::anyhow!("Need at least 2 matches for estimation"));
    }

    // Simple approach: use first two matches to estimate translation and rotation
    let m1 = sample_matches[0];
    let m2 = sample_matches[1];

    let pt1_1 = kp1[m1.query_idx as usize].pt();
    let pt2_1 = kp2[m1.train_idx as usize].pt();
    let pt1_2 = kp1[m2.query_idx as usize].pt();
    let pt2_2 = kp2[m2.train_idx as usize].pt();

    // Translation from first match
    let tx = pt2_1.x - pt1_1.x;
    let ty = pt2_1.y - pt1_1.y;

    // Rotation from vector between two points
    let dx1 = pt1_2.x - pt1_1.x;
    let dy1 = pt1_2.y - pt1_1.y;
    let dx2 = pt2_2.x - pt2_1.x;
    let dy2 = pt2_2.y - pt2_1.y;

    let angle1 = dy1.atan2(dx1);
    let angle2 = dy2.atan2(dx2);
    let rotation = (angle2 - angle1).to_degrees();

    // Scale from distance ratio
    let dist1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let dist2 = (dx2 * dx2 + dy2 * dy2).sqrt();
    let scale = if dist1 > 0.0 { dist2 / dist1 } else { 1.0 };

    Ok((tx, ty, rotation, scale))
}

/// Count inliers for a given transformation
fn count_inliers(
    kp1: &[KeyPoint],
    kp2: &[KeyPoint],
    matches: &[DMatch],
    transformation: &(f32, f32, f32, f32), // tx, ty, rotation, scale
    threshold: f32,
) -> usize {
    let (tx, ty, _rotation, _scale) = *transformation;
    
    matches
        .iter()
        .filter(|m| {
            let pt1 = kp1[m.query_idx as usize].pt();
            let pt2 = kp2[m.train_idx as usize].pt();

            // Predicted position of pt1 in image 2
            let predicted_x = pt1.x + tx;
            let predicted_y = pt1.y + ty;

            // Distance to actual position
            let dx = predicted_x - pt2.x;
            let dy = predicted_y - pt2.y;
            let distance = (dx * dx + dy * dy).sqrt();

            distance < threshold
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Point2f;

    fn create_test_keypoint(x: f32, y: f32, angle: f32) -> KeyPoint {
        KeyPoint::new_point(opencv::core::Point2f::new(x, y), 1.0, angle, 0.0, 0, -1).unwrap()
    }

    fn create_test_match(query_idx: i32, train_idx: i32, distance: f32) -> DMatch {
        DMatch::new(query_idx, train_idx, distance).unwrap()
    }

    #[test]
    fn test_few_matches_handling() {
        let kp1 = vec![
            create_test_keypoint(10.0, 10.0, 0.0),
            create_test_keypoint(20.0, 20.0, 0.0),
        ];
        let kp2 = vec![
            create_test_keypoint(15.0, 15.0, 0.0), // Translated by (5, 5)
            create_test_keypoint(25.0, 25.0, 0.0),
        ];
        let matches = vec![
            create_test_match(0, 0, 0.1),
            create_test_match(1, 1, 0.1),
        ];

        let result = handle_few_matches(&kp1, &kp2, &matches).unwrap();
        
        assert!((result.translation.0 - 5.0).abs() < 0.1);
        assert!((result.translation.1 - 5.0).abs() < 0.1);
        assert_eq!(result.total_matches, 2);
    }

    #[test]
    fn test_empty_matches() {
        let kp1 = vec![];
        let kp2 = vec![];
        let matches = vec![];

        let result = handle_few_matches(&kp1, &kp2, &matches).unwrap();
        
        assert_eq!(result.translation, (0.0, 0.0));
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.total_matches, 0);
    }
}