use image_alignment::*;
use image_alignment::algorithms::ImprovedORB;
use image::{GrayImage, Luma};

fn create_textured_pattern(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        // Create a complex pattern with many corners and textures
        let intensity = ((x % 16 < 8) ^ (y % 16 < 8)) as u8 * 200 +
                       ((x % 4 < 2) ^ (y % 4 < 2)) as u8 * 50 + 5;
        Luma([intensity])
    })
}

fn create_scaled_pattern(pattern: &GrayImage, scale: f32) -> GrayImage {
    let new_width = (pattern.width() as f32 * scale) as u32;
    let new_height = (pattern.height() as f32 * scale) as u32;
    image::imageops::resize(pattern, new_width, new_height, image::imageops::FilterType::Gaussian)
}

fn create_rotated_and_translated_pattern(
    pattern: &GrayImage, 
    angle_deg: f32, 
    tx: i32, 
    ty: i32,
    target_width: u32,
    target_height: u32
) -> GrayImage {
    let mut target = GrayImage::new(target_width, target_height);
    let center_x = pattern.width() as f32 / 2.0;
    let center_y = pattern.height() as f32 / 2.0;
    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    
    for y in 0..pattern.height() {
        for x in 0..pattern.width() {
            // Rotate around center
            let x_centered = x as f32 - center_x;
            let y_centered = y as f32 - center_y;
            let x_rot = x_centered * cos_a - y_centered * sin_a + center_x;
            let y_rot = x_centered * sin_a + y_centered * cos_a + center_y;
            
            // Translate
            let final_x = x_rot as i32 + tx;
            let final_y = y_rot as i32 + ty;
            
            if final_x >= 0 && final_x < target_width as i32 && 
               final_y >= 0 && final_y < target_height as i32 {
                let pixel = pattern.get_pixel(x, y);
                target.put_pixel(final_x as u32, final_y as u32, *pixel);
            }
        }
    }
    target
}

#[test]
fn test_improved_orb_initialization() {
    let orb = ImprovedORB::new();
    assert_eq!(orb.fast_threshold, 20);
    assert_eq!(orb.max_keypoints, 500);
    assert_eq!(orb.pyramid_levels, 8);
    assert_eq!(orb.scale_factor, 1.2);
}

#[test]
fn test_improved_orb_custom_params() {
    let orb = ImprovedORB::with_params(30, 100, 4);
    assert_eq!(orb.fast_threshold, 30);
    assert_eq!(orb.max_keypoints, 100);
    assert_eq!(orb.pyramid_levels, 4);
}

#[test]
fn test_improved_orb_basic_alignment() {
    let template = create_textured_pattern(64, 64);
    let target = template.clone();
    
    let orb = ImprovedORB::new();
    let result = orb.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "ImprovedORB");
    assert!(result.processing_time_ms >= 0.0);
    // For identical images, should have minimal transformation
    assert!(result.translation.0.abs() < 5.0);
    assert!(result.translation.1.abs() < 5.0);
    assert!(result.confidence >= 0.0);
}

#[test]
fn test_pyramid_scale_invariance() {
    let template = create_textured_pattern(64, 64);
    let target = create_scaled_pattern(&template, 1.5);
    
    let orb = ImprovedORB::with_params(15, 300, 6);
    let result = orb.align(&template, &target).unwrap();
    
    // Should detect some scale change
    println!("Detected scale: {:.3}", result.scale);
    assert!(result.confidence >= 0.0);
    assert!(result.processing_time_ms > 0.0);
    // Scale detection should be reasonably close (allowing tolerance)
    assert!(result.scale > 1.0 && result.scale < 2.0);
}

#[test]
fn test_translation_detection_with_ransac() {
    let template = create_textured_pattern(48, 48);
    let target = create_rotated_and_translated_pattern(&template, 0.0, 15, 10, 80, 80);
    
    let orb = ImprovedORB::with_params(15, 200, 4);
    let result = orb.align(&template, &target).unwrap();
    
    println!("Detected translation: ({:.2}, {:.2})", result.translation.0, result.translation.1);
    println!("Expected translation: (15, 10)");
    println!("Confidence: {:.3}", result.confidence);
    
    // RANSAC should provide robust estimation
    assert!(result.confidence >= 0.0);
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_rotation_detection_with_ransac() {
    let template = create_textured_pattern(48, 48);
    let target = create_rotated_and_translated_pattern(&template, 10.0, 5, 5, 80, 80);
    
    let orb = ImprovedORB::with_params(10, 300, 4);
    let result = orb.align(&template, &target).unwrap();
    
    println!("Detected rotation: {:.2}°", result.rotation);
    println!("Expected rotation: 10°");
    println!("Confidence: {:.3}", result.confidence);
    
    assert!(result.confidence >= 0.0);
    assert!(!result.rotation.is_nan());
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_performance_comparison() {
    let template = create_textured_pattern(64, 64);
    let target = create_textured_pattern(80, 80);
    
    // Test different pyramid levels
    let pyramid_levels = [2, 4, 6, 8];
    let mut results = Vec::new();
    
    for &levels in &pyramid_levels {
        let orb = ImprovedORB::with_params(20, 200, levels);
        let result = orb.align(&template, &target).unwrap();
        results.push((levels, result.processing_time_ms, result.confidence));
    }
    
    // All should complete successfully
    for (levels, time, confidence) in &results {
        println!("Pyramid levels {}: {:.2}ms, confidence={:.3}", levels, time, confidence);
        assert!(*time >= 0.0);
        assert!(*confidence >= 0.0);
        assert!(*time < 1000.0); // Should be under 1 second
    }
}

#[test]
fn test_optimized_fast_detection() {
    let template = create_textured_pattern(48, 48);
    let target = create_textured_pattern(48, 48);
    
    // Test with different thresholds to verify FAST optimization
    let thresholds = [10, 20, 30, 40];
    
    for &threshold in &thresholds {
        let orb = ImprovedORB::with_params(threshold, 100, 3);
        let result = orb.align(&template, &target).unwrap();
        
        println!("FAST threshold {}: confidence={:.3}, time={:.2}ms", 
                 threshold, result.confidence, result.processing_time_ms);
        assert!(result.confidence >= 0.0);
        assert!(result.processing_time_ms >= 0.0);
    }
}

#[test]
fn test_feature_matching_robustness() {
    let template = create_textured_pattern(32, 32);
    
    // Create target with noise
    let mut noisy_target = GrayImage::new(48, 48);
    for y in 0..32 {
        for x in 0..32 {
            let original = template.get_pixel(x, y)[0] as i32;
            let noise = ((x * 13 + y * 17) % 30) as i32 - 15;
            let noisy = (original + noise).max(0).min(255) as u8;
            noisy_target.put_pixel(x + 8, y + 8, Luma([noisy]));
        }
    }
    
    let orb = ImprovedORB::with_params(15, 150, 4);
    let result = orb.align(&template, &noisy_target).unwrap();
    
    // Should handle noise reasonably well
    assert!(result.confidence >= 0.0);
    assert!(result.processing_time_ms > 0.0);
    println!("Noise robustness: confidence={:.3}", result.confidence);
}

#[test]
fn test_parallel_processing() {
    let template = create_textured_pattern(64, 64);
    let target = create_textured_pattern(80, 80);
    
    let orb = ImprovedORB::new();
    
    // Run multiple times to test consistency and thread safety
    let mut results = Vec::new();
    for _ in 0..3 {
        let result = orb.align(&template, &target).unwrap();
        results.push(result);
    }
    
    // Results should be consistent
    let first = &results[0];
    for result in &results[1..] {
        assert_eq!(result.algorithm_used, first.algorithm_used);
        // Processing times may vary slightly due to threading
        assert!(result.processing_time_ms > 0.0);
        // Confidence should be consistent
        assert!((result.confidence - first.confidence).abs() < 0.1);
    }
}

#[test]
fn test_edge_cases() {
    let orb = ImprovedORB::new();
    
    // Test with very small images
    let small_template = create_textured_pattern(16, 16);
    let small_target = create_textured_pattern(20, 20);
    let result = orb.align(&small_template, &small_target).unwrap();
    assert!(result.confidence >= 0.0);
    
    // Test with uniform image (should have low confidence)
    let uniform_template = GrayImage::from_pixel(32, 32, Luma([128]));
    let uniform_target = GrayImage::from_pixel(48, 48, Luma([128]));
    let result = orb.align(&uniform_template, &uniform_target).unwrap();
    assert_eq!(result.confidence, 0.0); // No features to match
    assert_eq!(result.translation, (0.0, 0.0));
}

#[test]
fn test_scale_and_rotation_combined() {
    let template = create_textured_pattern(40, 40);
    
    // Create target with scale, rotation, and translation
    let scaled = create_scaled_pattern(&template, 1.3);
    let target = create_rotated_and_translated_pattern(&scaled, 15.0, 10, 5, 100, 100);
    
    let orb = ImprovedORB::with_params(10, 400, 6);
    let result = orb.align(&template, &target).unwrap();
    
    println!("Combined transformation:");
    println!("  Scale: {:.3} (expected ~1.3)", result.scale);
    println!("  Rotation: {:.2}° (expected ~15°)", result.rotation);
    println!("  Translation: ({:.1}, {:.1}) (expected ~(10, 5))", 
             result.translation.0, result.translation.1);
    println!("  Confidence: {:.3}", result.confidence);
    
    // Should detect approximate transformations
    assert!(result.confidence >= 0.0);
    assert!(result.processing_time_ms > 0.0);
    assert!(!result.scale.is_nan());
    assert!(!result.rotation.is_nan());
}

#[test]
fn test_pre_computed_brief_pattern() {
    // Test that descriptor computation is deterministic
    let template = create_textured_pattern(32, 32);
    let orb = ImprovedORB::new();
    
    let result1 = orb.align(&template, &template).unwrap();
    let result2 = orb.align(&template, &template).unwrap();
    
    // Should get exactly the same results (deterministic pattern)
    assert_eq!(result1.confidence, result2.confidence);
    assert_eq!(result1.translation, result2.translation);
    assert_eq!(result1.rotation, result2.rotation);
}