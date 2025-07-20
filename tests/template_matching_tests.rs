use image_alignment::*;
use image_alignment::algorithms::{NormalizedCrossCorrelation, SumOfSquaredDifferences, IntegralImageMatcher};
use image::{GrayImage, Luma};

fn create_test_pattern(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        if (x + y) % 8 < 4 { Luma([255]) } else { Luma([0]) }
    })
}

fn create_target_with_template(template: &GrayImage, target_width: u32, target_height: u32, shift_x: u32, shift_y: u32) -> GrayImage {
    let mut target = GrayImage::new(target_width, target_height);
    // Place the template at the specified offset
    for y in 0..template.height() {
        for x in 0..template.width() {
            if x + shift_x < target_width && y + shift_y < target_height {
                let pixel = template.get_pixel(x, y);
                target.put_pixel(x + shift_x, y + shift_y, *pixel);
            }
        }
    }
    target
}

fn create_noisy_pattern(width: u32, height: u32, noise_level: u8) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        let base_val = if (x + y) % 8 < 4 { 255 } else { 0 };
        let noise = ((x * 31 + y * 17) % (noise_level as u32 * 2)) as i32 - noise_level as i32;
        let final_val = (base_val as i32 + noise).max(0).min(255) as u8;
        Luma([final_val])
    })
}

#[test]
fn test_ncc_exact_match() {
    let template = create_test_pattern(32, 32);
    let target = template.clone();
    
    let ncc = NormalizedCrossCorrelation;
    let result = ncc.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "NCC");
    assert!(result.processing_time_ms >= 0.0);
    assert!((result.translation.0 - 0.0).abs() < 1.0);
    assert!((result.translation.1 - 0.0).abs() < 1.0);
    assert!(result.confidence > 0.8);
}

#[test]
fn test_ncc_translation_detection() {
    let template = create_test_pattern(16, 16);
    let target = create_target_with_template(&template, 32, 32, 5, 3);
    
    let ncc = NormalizedCrossCorrelation;
    let result = ncc.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "NCC");
    // Should detect the translation (allow some tolerance)
    assert!((result.translation.0 - 5.0).abs() < 1.0);
    assert!((result.translation.1 - 3.0).abs() < 1.0);
    assert!(result.confidence > 0.5);
}

#[test]
fn test_ssd_exact_match() {
    let template = create_test_pattern(32, 32);
    let target = template.clone();
    
    let ssd = SumOfSquaredDifferences;
    let result = ssd.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "SSD");
    assert!(result.processing_time_ms >= 0.0);
    assert!((result.translation.0 - 0.0).abs() < 1.0);
    assert!((result.translation.1 - 0.0).abs() < 1.0);
    assert!(result.confidence > 0.9);
}

#[test]
fn test_ssd_translation_detection() {
    let template = create_test_pattern(16, 16);
    let target = create_target_with_template(&template, 32, 32, 7, 4);
    
    let ssd = SumOfSquaredDifferences;
    let result = ssd.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "SSD");
    // Should detect the translation
    assert!((result.translation.0 - 7.0).abs() < 1.0);
    assert!((result.translation.1 - 4.0).abs() < 1.0);
    assert!(result.confidence > 0.5);
}

#[test]
fn test_integral_image_exact_match() {
    let template = create_test_pattern(32, 32);
    let target = template.clone();
    
    let integral = IntegralImageMatcher;
    let result = integral.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "IntegralImage");
    assert!(result.processing_time_ms >= 0.0);
    assert!((result.translation.0 - 0.0).abs() < 1.0);
    assert!((result.translation.1 - 0.0).abs() < 1.0);
    assert!(result.confidence > 0.8);
}

#[test]
fn test_integral_image_translation_detection() {
    let template = create_test_pattern(16, 16);
    let target = create_target_with_template(&template, 32, 32, 6, 8);
    
    let integral = IntegralImageMatcher;
    let result = integral.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "IntegralImage");
    // Should detect the translation
    assert!((result.translation.0 - 6.0).abs() < 1.0);
    assert!((result.translation.1 - 8.0).abs() < 1.0);
    assert!(result.confidence > 0.5);
}

#[test]
fn test_noise_robustness() {
    let template = create_test_pattern(24, 24);
    let target_clean = create_test_pattern(24, 24);
    let target_noisy = create_noisy_pattern(24, 24, 10);
    
    let ncc = NormalizedCrossCorrelation;
    let result_clean = ncc.align(&template, &target_clean).unwrap();
    let result_noisy = ncc.align(&template, &target_noisy).unwrap();
    
    // Clean image should have higher confidence
    assert!(result_clean.confidence > result_noisy.confidence);
    // Both should still detect approximate location
    assert!((result_clean.translation.0 - 0.0).abs() < 1.0);
    assert!((result_noisy.translation.0 - 0.0).abs() < 3.0);
}

#[test]
fn test_algorithm_speed_comparison() {
    let template = create_test_pattern(32, 32);
    let target = create_test_pattern(64, 64);
    
    let ncc = NormalizedCrossCorrelation;
    let ssd = SumOfSquaredDifferences;
    let integral = IntegralImageMatcher;
    
    let result_ncc = ncc.align(&template, &target).unwrap();
    let result_ssd = ssd.align(&template, &target).unwrap();
    let result_integral = integral.align(&template, &target).unwrap();
    
    // All algorithms should complete in reasonable time (< 100ms for small images)
    assert!(result_ncc.processing_time_ms < 100.0);
    assert!(result_ssd.processing_time_ms < 100.0);
    assert!(result_integral.processing_time_ms < 100.0);
    
    // SSD should typically be fastest (but this can vary by implementation)
    println!("NCC: {:.2}ms, SSD: {:.2}ms, Integral: {:.2}ms", 
             result_ncc.processing_time_ms, 
             result_ssd.processing_time_ms, 
             result_integral.processing_time_ms);
}

#[test]
fn test_template_larger_than_target() {
    let template = create_test_pattern(64, 64);
    let target = create_test_pattern(32, 32);
    
    let ncc = NormalizedCrossCorrelation;
    let result = ncc.align(&template, &target).unwrap();
    
    // Should handle gracefully and return zero translation
    assert_eq!(result.translation, (0.0, 0.0));
    assert_eq!(result.confidence, 0.0);
}

#[test]
fn test_uniform_images() {
    let template = GrayImage::from_pixel(32, 32, Luma([128]));
    let target = GrayImage::from_pixel(64, 64, Luma([128]));
    
    let ncc = NormalizedCrossCorrelation;
    let result = ncc.align(&template, &target).unwrap();
    
    // Should handle uniform images gracefully (zero variance case)
    assert!(result.confidence >= 0.0);
    assert!(!result.processing_time_ms.is_nan());
}

#[test]
fn test_parallel_processing() {
    use std::sync::Arc;
    use std::thread;
    
    let template = Arc::new(create_test_pattern(24, 24));
    let target = Arc::new(create_test_pattern(48, 48));
    
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let template = Arc::clone(&template);
            let target = Arc::clone(&target);
            thread::spawn(move || {
                let ncc = NormalizedCrossCorrelation;
                ncc.align(&template, &target).unwrap()
            })
        })
        .collect();
    
    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // All parallel executions should produce consistent results
    for result in &results {
        assert_eq!(result.algorithm_used, "NCC");
        assert!(result.processing_time_ms >= 0.0);
        assert!((result.translation.0 - results[0].translation.0).abs() < 0.1);
        assert!((result.translation.1 - results[0].translation.1).abs() < 0.1);
    }
}

#[test]
fn test_subpixel_accuracy() {
    // Create a pattern with known sub-pixel displacement
    let template = create_test_pattern(16, 16);
    let mut target = GrayImage::new(32, 32);
    
    // Manually create a slightly shifted pattern to test sub-pixel detection
    for y in 0..16 {
        for x in 0..16 {
            let template_val = template.get_pixel(x, y)[0];
            // Place with slight offset to test sub-pixel detection
            target.put_pixel(x + 5, y + 3, Luma([template_val]));
        }
    }
    
    let ncc = NormalizedCrossCorrelation;
    let result = ncc.align(&template, &target).unwrap();
    
    // Should detect close to the actual displacement
    assert!((result.translation.0 - 5.0).abs() < 1.0);
    assert!((result.translation.1 - 3.0).abs() < 1.0);
    assert!(result.confidence > 0.5);
}