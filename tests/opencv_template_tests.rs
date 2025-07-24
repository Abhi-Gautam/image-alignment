use image::{GrayImage, Luma};
use image_alignment::algorithms::*;

fn create_test_pattern(width: u32, height: u32, pattern_type: u8) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        match pattern_type {
            0 => Luma([((x + y) % 2 * 255) as u8]), // Checkerboard
            1 => Luma([((x % 8 < 4) ^ (y % 8 < 4)) as u8 * 255]), // Grid
            2 => Luma([((x % 4 == 0) || (y % 4 == 0)) as u8 * 255]), // Cross pattern
            _ => Luma([128]),                       // Gray
        }
    })
}

fn create_translated_image(original: &GrayImage, dx: i32, dy: i32) -> GrayImage {
    let width = original.width();
    let height = original.height();

    GrayImage::from_fn(width, height, |x, y| {
        let src_x = (x as i32 - dx).max(0).min(width as i32 - 1) as u32;
        let src_y = (y as i32 - dy).max(0).min(height as i32 - 1) as u32;
        original.get_pixel(src_x, src_y).clone()
    })
}

#[test]
fn test_ncc_exact_match() {
    let template = create_test_pattern(16, 16, 0);
    let target = create_test_pattern(16, 16, 0);

    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&template, &target).unwrap();

    assert_eq!(result.algorithm_used, "OpenCV-NCC");
    assert!(
        (result.translation.0.abs() < 1.0),
        "Translation X should be near 0, got {}",
        result.translation.0
    );
    assert!(
        (result.translation.1.abs() < 1.0),
        "Translation Y should be near 0, got {}",
        result.translation.1
    );
    assert!(
        result.confidence > 0.8,
        "Confidence should be high for exact match, got {}",
        result.confidence
    );
}

#[test]
fn test_ssd_exact_match() {
    let template = create_test_pattern(16, 16, 1);
    let target = create_test_pattern(16, 16, 1);

    let matcher = OpenCVTemplateMatcher::new_ssd();
    let result = matcher.align(&template, &target).unwrap();

    assert_eq!(result.algorithm_used, "OpenCV-SSD");
    assert!(
        (result.translation.0.abs() < 1.0),
        "Translation should be near 0"
    );
    assert!(
        (result.translation.1.abs() < 1.0),
        "Translation should be near 0"
    );
    assert!(
        result.confidence > 0.5,
        "Confidence should be reasonable for exact match"
    );
}

#[test]
fn test_ccorr_exact_match() {
    let template = create_test_pattern(16, 16, 2);
    let target = create_test_pattern(16, 16, 2);

    let matcher = OpenCVTemplateMatcher::new_ccorr();
    let result = matcher.align(&template, &target).unwrap();

    assert_eq!(result.algorithm_used, "OpenCV-CCORR");
    assert!(
        (result.translation.0.abs() < 1.0),
        "Translation should be near 0"
    );
    assert!(
        (result.translation.1.abs() < 1.0),
        "Translation should be near 0"
    );
    assert!(result.confidence > 0.0, "Confidence should be positive");
}

#[test]
fn test_ncc_translation_detection() {
    let template = create_test_pattern(32, 32, 0);
    let mut target = GrayImage::new(64, 64);

    // Place template at known offset in larger target
    let offset_x = 16;
    let offset_y = 12;

    for y in 0..template.height() {
        for x in 0..template.width() {
            let target_x = x + offset_x;
            let target_y = y + offset_y;
            let pixel = template.get_pixel(x, y);
            target.put_pixel(target_x, target_y, *pixel);
        }
    }

    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&template, &target).unwrap();

    // Expected translation from center of target to center of template placement
    let expected_x = offset_x as f32 + template.width() as f32 / 2.0 - target.width() as f32 / 2.0;
    let expected_y =
        offset_y as f32 + template.height() as f32 / 2.0 - target.height() as f32 / 2.0;

    assert!(
        (result.translation.0 - expected_x).abs() < 2.0,
        "Translation X should be near {}, got {}",
        expected_x,
        result.translation.0
    );
    assert!(
        (result.translation.1 - expected_y).abs() < 2.0,
        "Translation Y should be near {}, got {}",
        expected_y,
        result.translation.1
    );
    assert!(
        result.confidence > 0.7,
        "Should have high confidence for clear match"
    );
}

#[test]
fn test_template_larger_than_target() {
    let template = create_test_pattern(32, 32, 0);
    let target = create_test_pattern(16, 16, 0);

    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&template, &target);

    assert!(
        result.is_err(),
        "Should fail when template is larger than target"
    );
}

#[test]
fn test_uniform_images() {
    let template = GrayImage::from_pixel(16, 16, Luma([128]));
    let target = GrayImage::from_pixel(16, 16, Luma([128]));

    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&template, &target).unwrap();

    // Uniform images should still produce a result
    assert!(result.confidence >= 0.0);
    assert!(result.processing_time_ms > 0.0);
}

#[test]
fn test_noise_robustness() {
    let template = create_test_pattern(24, 24, 1);
    let mut target = template.clone();

    // Add some noise
    for y in 0..target.height() {
        for x in 0..target.width() {
            if (x + y) % 10 == 0 {
                let current = target.get_pixel(x, y)[0];
                let noisy = current.saturating_add(20);
                target.put_pixel(x, y, Luma([noisy]));
            }
        }
    }

    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&template, &target).unwrap();

    // Should still work reasonably well with noise
    assert!(
        result.translation.0.abs() < 3.0,
        "Should handle noise reasonably"
    );
    assert!(
        result.translation.1.abs() < 3.0,
        "Should handle noise reasonably"
    );
    assert!(
        result.confidence > 0.3,
        "Should have some confidence even with noise"
    );
}

#[test]
fn test_performance_comparison() {
    let template = create_test_pattern(20, 20, 0);
    let target = create_test_pattern(20, 20, 0);

    let algorithms = vec![
        OpenCVTemplateMatcher::new_ncc(),
        OpenCVTemplateMatcher::new_ssd(),
        OpenCVTemplateMatcher::new_ccorr(),
    ];

    let mut results = Vec::new();

    for algorithm in algorithms {
        let result = algorithm.align(&template, &target).unwrap();
        results.push(result);
    }

    // All algorithms should complete in reasonable time
    for result in &results {
        assert!(
            result.processing_time_ms < 1000.0,
            "Algorithm {} too slow: {}ms",
            result.algorithm_used,
            result.processing_time_ms
        );
    }

    // Check that all algorithms produce reasonable results
    for result in &results {
        assert!(result.translation.0.abs() < 2.0);
        assert!(result.translation.1.abs() < 2.0);
    }
}

#[test]
fn test_sub_pixel_accuracy() {
    let template = create_test_pattern(16, 16, 2);
    let target = create_translated_image(&template, 1, 1);

    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&template, &target).unwrap();

    // Should detect the 1-pixel translation
    assert!(
        (result.translation.0 - 1.0).abs() < 1.5,
        "Should detect translation close to 1.0, got {}",
        result.translation.0
    );
    assert!(
        (result.translation.1 - 1.0).abs() < 1.5,
        "Should detect translation close to 1.0, got {}",
        result.translation.1
    );
}

#[test]
fn test_edge_cases() {
    let small_template = create_test_pattern(8, 8, 0);
    let large_target = create_test_pattern(64, 64, 0);

    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&small_template, &large_target).unwrap();

    // Should handle size differences gracefully
    assert!(result.processing_time_ms > 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_different_patterns() {
    let template_patterns = vec![0, 1, 2];
    let target_patterns = vec![0, 1, 2];

    for &template_pattern in &template_patterns {
        for &target_pattern in &target_patterns {
            let template = create_test_pattern(16, 16, template_pattern);
            let target = create_test_pattern(16, 16, target_pattern);

            let matcher = OpenCVTemplateMatcher::new_ncc();
            let result = matcher.align(&template, &target).unwrap();

            if template_pattern == target_pattern {
                // Same patterns should have high confidence
                assert!(
                    result.confidence > 0.5,
                    "Same pattern match should have confidence > 0.5, got {}",
                    result.confidence
                );
            }

            // All should run without error
            assert!(result.processing_time_ms > 0.0);
        }
    }
}
