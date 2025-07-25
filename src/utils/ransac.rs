use crate::config::RansacConfig;
use crate::Result;
use opencv::core::{DMatch, KeyPoint};
use opencv::prelude::*;
use rand::seq::SliceRandom;


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
    config: Option<&RansacConfig>,
) -> Result<RansacResult> {
    let default_config = RansacConfig::default();
    let config = config.unwrap_or(&default_config);

    if matches.len() < config.min_inliers as usize {
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
        let sample_matches: Vec<&DMatch> = sample_indices.iter().map(|&i| &matches[i]).collect();

        if let Ok(transformation) = estimate_from_sample(kp1, kp2, &sample_matches) {
            // Count inliers for this transformation
            let inlier_count =
                count_inliers(kp1, kp2, matches, &transformation, config.inlier_threshold);

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
    let confidence = if best_inlier_count >= config.min_inliers as usize {
        (inlier_ratio * 0.7 + 0.3).min(1.0) // Scale to [0.3, 1.0] - less conservative
    } else {
        0.3 // Higher minimum confidence for few matches
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
    let mut scale_sum = 0.0;
    let mut valid_scales = 0;

    for m in matches {
        let pt1 = kp1[m.query_idx as usize].pt();
        let pt2 = kp2[m.train_idx as usize].pt();

        dx_sum += pt2.x - pt1.x;
        dy_sum += pt2.y - pt1.y;

        // Skip angle calculation in handle_few_matches as keypoint angles
        // are not reliable for rotation estimation between patches
        // Use 0 rotation as default assumption for few matches

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
    let rotation = 0.0; // Assume no rotation for few matches to avoid false positives
    let scale = if valid_scales > 0 {
        scale_sum / valid_scales as f32
    } else {
        1.0
    };

    let confidence = 0.4; // Improved confidence for few matches

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

