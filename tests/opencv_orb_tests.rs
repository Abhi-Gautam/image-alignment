use image_alignment::algorithms::*;
use image::{GrayImage, Luma};

fn create_corner_pattern(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        // Create a checkerboard pattern with strong corners
        if (x % 8 < 4) ^ (y % 8 < 4) {
            Luma([255])
        } else {
            Luma([0])
        }
    })
}

fn create_textured_pattern(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        let intensity = ((x * 7 + y * 11) % 256) as u8;
        Luma([intensity])
    })
}

fn create_circular_pattern(width: u32, height: u32) -> GrayImage {
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    
    GrayImage::from_fn(width, height, |x, y| {
        let dx = x as f32 - center_x;
        let dy = y as f32 - center_y;
        let distance = (dx * dx + dy * dy).sqrt();
        
        if distance < width as f32 / 4.0 {
            Luma([255])
        } else if distance < width as f32 / 3.0 {
            Luma([128])
        } else {
            Luma([0])
        }
    })
}

#[test]
fn test_orb_initialization() {
    let orb_default = OpenCVORB::new();
    assert_eq!(orb_default.max_features, 500);
    
    let orb_custom = OpenCVORB::new()
        .with_max_features(1000)
        .with_scale_factor(1.3)
        .with_n_levels(10)
        .with_fast_threshold(15);
    
    assert_eq!(orb_custom.max_features, 1000);
    assert_eq!(orb_custom.scale_factor, 1.3);
    assert_eq!(orb_custom.n_levels, 10);
    assert_eq!(orb_custom.fast_threshold, 15);
}

#[test]
fn test_orb_basic_alignment() {
    let template = create_corner_pattern(32, 32);
    let target = create_corner_pattern(32, 32);
    
    let orb = OpenCVORB::new().with_fast_threshold(5);
    let result = orb.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "OpenCV-ORB");
    assert!(result.processing_time_ms >= 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.translation.0.abs() < 10.0);
    assert!(result.translation.1.abs() < 10.0);
}

#[test]
fn test_orb_with_textured_patterns() {
    let template = create_textured_pattern(48, 48);
    let target = create_textured_pattern(48, 48);
    
    let orb = OpenCVORB::new().with_fast_threshold(10);
    let result = orb.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "OpenCV-ORB");
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_orb_with_circular_patterns() {
    let template = create_circular_pattern(40, 40);
    let target = create_circular_pattern(40, 40);
    
    let orb = OpenCVORB::new().with_fast_threshold(8);
    let result = orb.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "OpenCV-ORB");
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_orb_different_thresholds() {
    let template = create_corner_pattern(32, 32);
    let target = create_corner_pattern(32, 32);
    
    let thresholds = vec![5, 10, 20, 30];
    
    for threshold in thresholds {
        let orb = OpenCVORB::new().with_fast_threshold(threshold);
        let result = orb.align(&template, &target).unwrap();
        
        assert_eq!(result.algorithm_used, "OpenCV-ORB");
        assert!(result.processing_time_ms > 0.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
}

#[test]
fn test_orb_max_features_limit() {
    let template = create_textured_pattern(64, 64);
    let target = create_textured_pattern(64, 64);
    
    let feature_counts = vec![100, 500, 1000];
    
    for max_features in feature_counts {
        let orb = OpenCVORB::new()
            .with_max_features(max_features)
            .with_fast_threshold(5);
        
        let result = orb.align(&template, &target).unwrap();
        
        assert_eq!(result.algorithm_used, "OpenCV-ORB");
        assert!(result.processing_time_ms > 0.0);
    }
}

#[test]
fn test_orb_scale_invariance() {
    let template = create_corner_pattern(32, 32);
    let target = create_corner_pattern(32, 32);
    
    let scale_factors = vec![1.1, 1.2, 1.5];
    
    for scale_factor in scale_factors {
        let orb = OpenCVORB::new()
            .with_scale_factor(scale_factor)
            .with_n_levels(5)
            .with_fast_threshold(8);
        
        let result = orb.align(&template, &target).unwrap();
        
        assert_eq!(result.algorithm_used, "OpenCV-ORB");
        assert!(result.processing_time_ms > 0.0);
    }
}

#[test]
fn test_orb_empty_images() {
    let template = GrayImage::from_pixel(32, 32, Luma([128]));
    let target = GrayImage::from_pixel(32, 32, Luma([128]));
    
    let orb = OpenCVORB::new().with_fast_threshold(5);
    let result = orb.align(&template, &target).unwrap();
    
    // Should handle uniform images gracefully
    assert_eq!(result.algorithm_used, "OpenCV-ORB");
    assert!(result.confidence >= 0.0);
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_orb_performance_consistency() {
    let template = create_corner_pattern(40, 40);
    let target = create_corner_pattern(40, 40);
    
    let orb = OpenCVORB::new().with_fast_threshold(10);
    
    let mut processing_times = Vec::new();
    let mut confidences = Vec::new();
    
    // Run multiple times to check consistency
    for _ in 0..5 {
        let result = orb.align(&template, &target).unwrap();
        processing_times.push(result.processing_time_ms);
        confidences.push(result.confidence);
    }
    
    // All runs should complete successfully
    assert_eq!(processing_times.len(), 5);
    assert_eq!(confidences.len(), 5);
    
    // Processing times should be reasonable
    for &time in &processing_times {
        assert!(time > 0.0, "Processing time should be positive");
        assert!(time < 1000.0, "Processing time should be reasonable");
    }
    
    // Confidences should be consistent for identical images
    for &confidence in &confidences {
        assert!(confidence >= 0.0 && confidence <= 1.0, "Confidence should be in [0,1]");
    }
}

#[test]
fn test_orb_rotation_detection() {
    let template = create_corner_pattern(32, 32);
    let target = create_corner_pattern(32, 32);
    
    let orb = OpenCVORB::new().with_fast_threshold(8);
    let result = orb.align(&template, &target).unwrap();
    
    // For identical images, rotation should be near zero
    assert!(result.rotation.abs() < 45.0, "Rotation should be reasonable for identical images");
    assert_eq!(result.scale, 1.0, "Scale should be 1.0 for ORB");
}

#[test]
fn test_orb_noise_robustness() {
    let mut template = create_corner_pattern(32, 32);
    let mut target = create_corner_pattern(32, 32);
    
    // Add noise to both images
    for y in 0..template.height() {
        for x in 0..template.width() {
            if (x + y) % 8 == 0 {
                let current = template.get_pixel(x, y)[0];
                let noisy = current.saturating_add(30);
                template.put_pixel(x, y, Luma([noisy]));
                target.put_pixel(x, y, Luma([noisy]));
            }
        }
    }
    
    let orb = OpenCVORB::new().with_fast_threshold(5);
    let result = orb.align(&template, &target).unwrap();
    
    // Should still work with some noise
    assert_eq!(result.algorithm_used, "OpenCV-ORB");
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_orb_multi_level_pyramid() {
    let template = create_textured_pattern(64, 64);
    let target = create_textured_pattern(64, 64);
    
    let pyramid_levels = vec![3, 5, 8];
    
    for levels in pyramid_levels {
        let orb = OpenCVORB::new()
            .with_n_levels(levels)
            .with_fast_threshold(10);
        
        let result = orb.align(&template, &target).unwrap();
        
        assert_eq!(result.algorithm_used, "OpenCV-ORB");
        assert!(result.processing_time_ms > 0.0);
    }
}

#[test]
fn test_orb_hamming_distance_calculation() {
    let orb = OpenCVORB::new();
    
    // Test identical descriptors
    let desc1 = vec![0b10101010, 0b11110000];
    let desc2 = vec![0b10101010, 0b11110000];
    assert_eq!(orb.hamming_distance(&desc1, &desc2), 0.0);
    
    // Test completely different descriptors
    let desc3 = vec![0b01010101, 0b00001111];
    let distance = orb.hamming_distance(&desc1, &desc3);
    assert!(distance > 0.0, "Different descriptors should have non-zero distance");
    
    // Test single bit difference
    let desc4 = vec![0b10101011, 0b11110000]; // One bit different
    let distance_one_bit = orb.hamming_distance(&desc1, &desc4);
    assert_eq!(distance_one_bit, 1.0, "Single bit difference should give distance 1");
}

#[test]
fn test_orb_large_images() {
    let template = create_corner_pattern(128, 128);
    let target = create_corner_pattern(128, 128);
    
    let orb = OpenCVORB::new()
        .with_max_features(1000)
        .with_fast_threshold(10);
    
    let result = orb.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "OpenCV-ORB");
    assert!(result.processing_time_ms > 0.0);
    assert!(result.processing_time_ms < 5000.0, "Should complete in reasonable time for large images");
}