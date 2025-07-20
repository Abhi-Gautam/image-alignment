use image_alignment::*;
use image::{GrayImage, Luma};
use tempfile::tempdir;

#[test]
fn test_image_loading() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.png");
    
    // Create a simple test image
    let img = GrayImage::from_fn(32, 32, |x, y| {
        if (x + y) % 2 == 0 { Luma([255]) } else { Luma([0]) }
    });
    img.save(&file_path).unwrap();
    
    // Test loading
    let loaded = load_image(&file_path).unwrap();
    assert_eq!(loaded.width(), 32);
    assert_eq!(loaded.height(), 32);
}

#[test]
fn test_image_validation() {
    let small_img = GrayImage::new(4, 4);
    let large_img = GrayImage::new(64, 64);
    
    assert!(validate_image_size(&small_img, 8).is_err());
    assert!(validate_image_size(&large_img, 8).is_ok());
}

fn create_test_pattern(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        if (x + y) % 8 < 4 { Luma([255]) } else { Luma([0]) }
    })
}

#[test]
fn test_simple_pattern_creation() {
    let pattern = create_test_pattern(32, 32);
    assert_eq!(pattern.width(), 32);
    assert_eq!(pattern.height(), 32);
}

#[test]
fn test_opencv_template_ncc_alignment() {
    let template = create_test_pattern(32, 32);
    let target = create_test_pattern(32, 32);
    
    let matcher = OpenCVTemplateMatcher::new_ncc();
    let result = matcher.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "OpenCV-NCC");
    assert!(result.processing_time_ms >= 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_opencv_template_ssd_alignment() {
    let template = create_test_pattern(32, 32);
    let target = create_test_pattern(32, 32);
    
    let matcher = OpenCVTemplateMatcher::new_ssd();
    let result = matcher.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "OpenCV-SSD");
    assert!(result.processing_time_ms >= 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_opencv_orb_alignment() {
    let template = create_test_pattern(32, 32);
    let target = create_test_pattern(32, 32);
    
    let orb = OpenCVORB::new();
    let result = orb.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "OpenCV-ORB");
    assert!(result.processing_time_ms >= 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_phase_correlation_alignment() {
    let template = create_test_pattern(32, 32);
    let target = create_test_pattern(32, 32);
    
    let phase = PhaseCorrelation;
    let result = phase.align(&template, &target).unwrap();
    
    assert_eq!(result.algorithm_used, "PhaseCorrelation");
    assert!(result.processing_time_ms >= 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_metrics_calculation() {
    let result = AlignmentResult {
        translation: (5.0, 3.0),
        rotation: 10.0,
        scale: 1.1,
        confidence: 0.8,
        processing_time_ms: 15.0,
        algorithm_used: "Test".to_string(),
    };
    
    let translation_error = calculate_translation_error(&result, (2.0, 1.0));
    assert!((translation_error - ((3.0_f32).powi(2) + (2.0_f32).powi(2)).sqrt()).abs() < 0.001);
    
    let rotation_error = calculate_rotation_error(&result, 5.0);
    assert!((rotation_error - 5.0).abs() < 0.001);
    
    let scale_error = calculate_scale_error(&result, 1.0);
    assert!((scale_error - 0.1).abs() < 0.001);
}

#[test]
fn test_benchmark_runner() {
    let template = create_test_pattern(16, 16);
    let target = create_test_pattern(16, 16);
    
    let mut runner = BenchmarkRunner::new();
    runner.add_algorithm(Box::new(OpenCVTemplateMatcher::new_ncc()));
    runner.add_algorithm(Box::new(OpenCVORB::new()));
    runner.add_algorithm(Box::new(PhaseCorrelation));
    
    let results = runner.run_benchmark(&template, &target);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].algorithm_used, "OpenCV-NCC");
    assert_eq!(results[1].algorithm_used, "OpenCV-ORB");
    assert_eq!(results[2].algorithm_used, "PhaseCorrelation");
}

#[test]
fn test_algorithm_comparison() {
    let template = create_test_pattern(24, 24);
    let target = create_test_pattern(24, 24);
    
    // Test multiple OpenCV algorithms
    let algorithms: Vec<Box<dyn AlignmentAlgorithm>> = vec![
        Box::new(OpenCVTemplateMatcher::new_ncc()),
        Box::new(OpenCVTemplateMatcher::new_ssd()),
        Box::new(OpenCVTemplateMatcher::new_ccorr()),
        Box::new(OpenCVORB::new()),
    ];
    
    for algorithm in algorithms {
        let result = algorithm.align(&template, &target).unwrap();
        
        // All algorithms should run successfully
        assert!(result.processing_time_ms >= 0.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        
        // For identical images, translation should be near zero
        assert!(result.translation.0.abs() < 5.0, 
               "Algorithm {} translation X too large: {}", 
               result.algorithm_used, result.translation.0);
        assert!(result.translation.1.abs() < 5.0, 
               "Algorithm {} translation Y too large: {}", 
               result.algorithm_used, result.translation.1);
    }
}