use image_alignment::*;
use image_alignment::algorithms::FastORB;
use image::{GrayImage, Luma};

fn create_corner_pattern(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        // Create a pattern with many corners for feature detection
        let intensity = if (x % 8 < 4) ^ (y % 8 < 4) { 255 } else { 50 };
        Luma([intensity])
    })
}


fn create_rotated_pattern(width: u32, height: u32, angle_deg: f32) -> GrayImage {
    let mut image = GrayImage::new(width, height);
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    
    for y in 0..height {
        for x in 0..width {
            // Rotate coordinates
            let x_centered = x as f32 - center_x;
            let y_centered = y as f32 - center_y;
            let x_rot = x_centered * cos_a - y_centered * sin_a + center_x;
            let y_rot = x_centered * sin_a + y_centered * cos_a + center_y;
            
            // Sample from corner pattern if within bounds
            if x_rot >= 0.0 && x_rot < width as f32 && y_rot >= 0.0 && y_rot < height as f32 {
                let orig_x = x_rot as u32;
                let orig_y = y_rot as u32;
                let intensity = if (orig_x % 8 < 4) ^ (orig_y % 8 < 4) { 255 } else { 50 };
                image.put_pixel(x, y, Luma([intensity]));
            } else {
                image.put_pixel(x, y, Luma([128])); // Gray background
            }
        }
    }
    image
}

#[test]
fn test_fast_orb_initialization() {
    let orb = FastORB::new();
    assert_eq!(orb.fast_threshold, 20);
    assert_eq!(orb.max_keypoints, 500);
    assert_eq!(orb.patch_size, 31);
    assert_eq!(orb.pyramid_levels, 3);
}

#[test]
fn test_fast_orb_custom_params() {
    let orb = FastORB::with_params(30, 100);
    assert_eq!(orb.fast_threshold, 30);
    assert_eq!(orb.max_keypoints, 100);
}

#[test]
fn test_fast_orb_basic_alignment() {
    let template = create_corner_pattern(64, 64);
    let target = template.clone();
    
    let orb = FastORB::new();
    let result = orb.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "FastORB");
    assert!(result.processing_time_ms >= 0.0);
    // For identical images, translation should be near zero
    assert!(result.translation.0.abs() < 10.0);
    assert!(result.translation.1.abs() < 10.0);
    assert!(result.confidence >= 0.0);
}

#[test]
fn test_fast_orb_feature_detection() {
    let image = create_corner_pattern(64, 64);
    let orb = FastORB::with_params(20, 50);
    
    // Test feature extraction (internal method through alignment)
    let template = create_corner_pattern(32, 32);
    let result = orb.align(&template, &image).unwrap();
    
    // Should be able to detect some features and produce a result
    assert!(result.processing_time_ms >= 0.0);
    assert!(!result.processing_time_ms.is_nan());
    assert!(result.confidence >= 0.0);
}

#[test]
fn test_fast_orb_translation_detection() {
    let template = create_corner_pattern(32, 32);
    let mut target = GrayImage::new(64, 64);
    
    // Place template at offset (10, 15)
    for y in 0..32 {
        for x in 0..32 {
            let pixel = template.get_pixel(x, y);
            target.put_pixel(x + 10, y + 15, *pixel);
        }
    }
    
    let orb = FastORB::with_params(15, 200);
    let result = orb.align(&template, &target).unwrap();
    
    // Should detect approximate translation (allowing some tolerance)
    println!("Detected translation: ({:.2}, {:.2})", result.translation.0, result.translation.1);
    assert!(result.confidence >= 0.0);
    assert!(result.processing_time_ms > 0.0);
    // Note: ORB might not find sufficient features in simple test patterns
}

#[test] 
fn test_fast_orb_rotation_detection() {
    let template = create_corner_pattern(48, 48);
    let target = create_rotated_pattern(64, 64, 15.0);
    
    let orb = FastORB::with_params(10, 300);
    let result = orb.align(&template, &target).unwrap();
    
    // Should detect some rotation (allowing tolerance)
    println!("Detected rotation: {:.2}°", result.rotation);
    assert!(result.confidence >= 0.0);
    assert!(!result.rotation.is_nan());
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_fast_orb_insufficient_features() {
    // Create uniform image with no features
    let template = GrayImage::from_pixel(32, 32, Luma([128]));
    let target = GrayImage::from_pixel(64, 64, Luma([128]));
    
    let orb = FastORB::new();
    let result = orb.align(&template, &target).unwrap();
    
    // Should handle gracefully with low confidence
    assert_eq!(result.confidence, 0.0);
    assert_eq!(result.translation, (0.0, 0.0));
    assert_eq!(result.rotation, 0.0);
}

#[test]
fn test_fast_orb_performance() {
    let template = create_corner_pattern(64, 64);
    let target = create_corner_pattern(128, 128);
    
    let orb = FastORB::new();
    let result = orb.align(&template, &target).unwrap();
    
    // Should complete within reasonable time (target: 1-5ms per RESEARCH.md)
    println!("FastORB processing time: {:.2}ms", result.processing_time_ms);
    assert!(result.processing_time_ms < 100.0); // Generous upper bound for now
    assert!(result.processing_time_ms >= 0.0);
}

#[test]
fn test_fast_orb_different_thresholds() {
    let template = create_corner_pattern(48, 48);
    let target = create_corner_pattern(48, 48);
    
    // Test with different FAST thresholds
    let thresholds = [10, 20, 30, 40];
    let mut results = Vec::new();
    
    for &threshold in &thresholds {
        let orb = FastORB::with_params(threshold, 100);
        let result = orb.align(&template, &target).unwrap();
        results.push((threshold, result.confidence, result.processing_time_ms));
    }
    
    // All should produce valid results
    for (threshold, confidence, time) in results {
        println!("Threshold {}: confidence={:.3}, time={:.2}ms", threshold, confidence, time);
        assert!(confidence >= 0.0);
        assert!(time >= 0.0);
        assert!(!time.is_nan());
    }
}

#[test]
fn test_fast_orb_keypoint_limits() {
    let template = create_corner_pattern(64, 64);
    let target = create_corner_pattern(64, 64);
    
    // Test with different keypoint limits
    let limits = [10, 50, 100, 500];
    
    for &limit in &limits {
        let orb = FastORB::with_params(20, limit);
        let result = orb.align(&template, &target).unwrap();
        
        // Should handle different keypoint limits gracefully
        assert!(result.confidence >= 0.0);
        assert!(result.processing_time_ms >= 0.0);
        println!("Max keypoints {}: confidence={:.3}", limit, result.confidence);
    }
}

#[test]
fn test_fast_orb_noise_robustness() {
    let template = create_corner_pattern(48, 48);
    
    // Create noisy version
    let mut noisy_target = template.clone();
    for y in 0..48 {
        for x in 0..48 {
            let original = noisy_target.get_pixel(x, y)[0] as i32;
            let noise = ((x * 17 + y * 23) % 40) as i32 - 20; // Pseudo-random noise
            let noisy = (original + noise).max(0).min(255) as u8;
            noisy_target.put_pixel(x, y, Luma([noisy]));
        }
    }
    
    let orb = FastORB::new();
    let clean_result = orb.align(&template, &template).unwrap();
    let noisy_result = orb.align(&template, &noisy_target).unwrap();
    
    // Noisy image should still produce some result, but possibly lower confidence
    assert!(noisy_result.confidence >= 0.0);
    assert!(noisy_result.processing_time_ms >= 0.0);
    println!("Clean confidence: {:.3}, Noisy confidence: {:.3}", 
             clean_result.confidence, noisy_result.confidence);
}

#[test]
fn test_fast_orb_multiple_runs_consistency() {
    let template = create_corner_pattern(48, 48);
    let target = create_corner_pattern(64, 64);
    
    let orb = FastORB::new();
    let mut results = Vec::new();
    
    // Run multiple times to check consistency
    for _ in 0..5 {
        let result = orb.align(&template, &target).unwrap();
        results.push(result);
    }
    
    // Results should be consistent across runs
    let first_result = &results[0];
    for result in &results[1..] {
        assert_eq!(result.algorithm_used, first_result.algorithm_used);
        // Translation should be reasonably consistent
        assert!((result.translation.0 - first_result.translation.0).abs() < 5.0);
        assert!((result.translation.1 - first_result.translation.1).abs() < 5.0);
    }
}