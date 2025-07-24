use image::{GrayImage, Luma};
use image_alignment::algorithms::*;
use image_alignment::*;
use std::time::Instant;

fn create_test_pattern(width: u32, height: u32, pattern_type: u8) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        match pattern_type {
            0 => Luma([((x + y) % 2 * 255) as u8]), // Checkerboard
            1 => Luma([((x % 8 < 4) ^ (y % 8 < 4)) as u8 * 255]), // Grid
            2 => Luma([((x * 7 + y * 11) % 256) as u8]), // Textured
            _ => Luma([128]),                       // Gray
        }
    })
}

fn benchmark_algorithm(
    algorithm: &dyn AlignmentAlgorithm,
    template: &GrayImage,
    target: &GrayImage,
    iterations: usize,
) -> (f32, f32, f32) {
    let mut total_time = 0.0;
    let mut total_confidence = 0.0;
    let mut successful_runs = 0;

    for _ in 0..iterations {
        let start = Instant::now();
        if let Ok(result) = algorithm.align(template, target) {
            let duration = start.elapsed().as_secs_f32() * 1000.0;
            total_time += duration;
            total_confidence += result.confidence;
            successful_runs += 1;
        }
    }

    if successful_runs > 0 {
        (
            total_time / successful_runs as f32,
            total_confidence / successful_runs as f32,
            successful_runs as f32 / iterations as f32,
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

#[test]
fn test_template_matching_performance() {
    let template = create_test_pattern(32, 32, 0);
    let target = create_test_pattern(32, 32, 0);
    let iterations = 10;

    let algorithms: Vec<Box<dyn AlignmentAlgorithm>> = vec![
        Box::new(OpenCVTemplateMatcher::new_ncc()),
        Box::new(OpenCVTemplateMatcher::new_ssd()),
        Box::new(OpenCVTemplateMatcher::new_ccorr()),
    ];

    for algorithm in algorithms {
        let (avg_time, avg_confidence, success_rate) =
            benchmark_algorithm(algorithm.as_ref(), &template, &target, iterations);

        println!(
            "{}: {:.2}ms avg, {:.3} confidence, {:.1}% success",
            algorithm.name(),
            avg_time,
            avg_confidence,
            success_rate * 100.0
        );

        // Performance requirements
        assert!(
            avg_time < 500.0,
            "Algorithm {} too slow: {:.2}ms",
            algorithm.name(),
            avg_time
        );
        assert!(
            success_rate > 0.8,
            "Algorithm {} success rate too low: {:.1}%",
            algorithm.name(),
            success_rate * 100.0
        );
        assert!(
            avg_confidence > 0.0,
            "Algorithm {} confidence too low: {:.3}",
            algorithm.name(),
            avg_confidence
        );
    }
}

#[test]
fn test_orb_performance() {
    let template = create_test_pattern(48, 48, 1);
    let target = create_test_pattern(48, 48, 1);
    let iterations = 5; // ORB is more complex, fewer iterations

    let orb_configs = vec![
        OpenCVORB::new()
            .with_fast_threshold(10)
            .with_max_features(100),
        OpenCVORB::new()
            .with_fast_threshold(10)
            .with_max_features(500),
        OpenCVORB::new()
            .with_fast_threshold(5)
            .with_max_features(500),
        OpenCVORB::new()
            .with_fast_threshold(15)
            .with_max_features(500),
    ];

    for orb in orb_configs {
        let (avg_time, avg_confidence, success_rate) =
            benchmark_algorithm(&orb, &template, &target, iterations);

        println!(
            "ORB(thresh={}, max={}): {:.2}ms avg, {:.3} confidence, {:.1}% success",
            orb.fast_threshold,
            orb.max_features,
            avg_time,
            avg_confidence,
            success_rate * 100.0
        );

        // ORB performance requirements (more lenient due to complexity)
        assert!(avg_time < 2000.0, "ORB too slow: {:.2}ms", avg_time);
        assert!(
            success_rate > 0.6,
            "ORB success rate too low: {:.1}%",
            success_rate * 100.0
        );
    }
}

#[test]
fn test_cross_algorithm_performance() {
    let template = create_test_pattern(40, 40, 2);
    let target = create_test_pattern(40, 40, 2);
    let iterations = 5;

    let mut runner = BenchmarkRunner::new();
    runner.add_algorithm(Box::new(OpenCVTemplateMatcher::new_ncc()));
    runner.add_algorithm(Box::new(OpenCVTemplateMatcher::new_ssd()));
    runner.add_algorithm(Box::new(OpenCVORB::new().with_fast_threshold(10)));
    runner.add_algorithm(Box::new(PhaseCorrelation));

    let mut all_times = Vec::new();

    for _ in 0..iterations {
        let results = runner.run_benchmark(&template, &target);

        for result in results {
            println!(
                "{}: {:.2}ms, confidence: {:.3}, translation: ({:.1}, {:.1})",
                result.algorithm_used,
                result.processing_time_ms,
                result.confidence,
                result.translation.0,
                result.translation.1
            );

            all_times.push((result.algorithm_used.clone(), result.processing_time_ms));
        }
    }

    // Check that all algorithms complete in reasonable time
    for (name, time) in all_times {
        assert!(time < 2000.0, "Algorithm {} too slow: {:.2}ms", name, time);
        assert!(
            time >= 0.0,
            "Algorithm {} invalid time: {:.2}ms",
            name,
            time
        );
    }
}

#[test]
fn test_accuracy_vs_performance_tradeoff() {
    let template = create_test_pattern(32, 32, 1);
    let target = template.clone(); // Exact match for accuracy testing

    // Test different ORB configurations
    let configs = vec![
        (
            "Fast",
            OpenCVORB::new()
                .with_fast_threshold(20)
                .with_max_features(100),
        ),
        (
            "Balanced",
            OpenCVORB::new()
                .with_fast_threshold(10)
                .with_max_features(500),
        ),
        (
            "Accurate",
            OpenCVORB::new()
                .with_fast_threshold(5)
                .with_max_features(1000),
        ),
    ];

    for (config_name, orb) in configs {
        let (avg_time, avg_confidence, success_rate) =
            benchmark_algorithm(&orb, &template, &target, 3);

        println!(
            "{}: {:.2}ms, {:.3} confidence, {:.1}% success",
            config_name,
            avg_time,
            avg_confidence,
            success_rate * 100.0
        );

        // All configs should work
        assert!(success_rate > 0.5, "{} success rate too low", config_name);
        assert!(
            avg_time > 0.0,
            "{} should have positive processing time",
            config_name
        );
    }
}

#[test]
fn test_image_size_scaling() {
    let sizes = vec![16, 32, 64];
    let template_matcher = OpenCVTemplateMatcher::new_ncc();

    for size in sizes {
        let template = create_test_pattern(size, size, 0);
        let target = create_test_pattern(size, size, 0);

        let (avg_time, avg_confidence, success_rate) =
            benchmark_algorithm(&template_matcher, &template, &target, 3);

        println!(
            "Size {}x{}: {:.2}ms, {:.3} confidence, {:.1}% success",
            size,
            size,
            avg_time,
            avg_confidence,
            success_rate * 100.0
        );

        // Performance should scale reasonably with image size
        let expected_max_time = (size as f32 / 16.0).powi(2) * 100.0; // Quadratic scaling estimate
        assert!(
            avg_time < expected_max_time,
            "Size {} performance worse than expected: {:.2}ms vs {:.2}ms max",
            size,
            avg_time,
            expected_max_time
        );

        assert!(success_rate > 0.8, "Size {} success rate too low", size);
    }
}

#[test]
fn test_memory_efficiency() {
    // Test with larger images to check memory usage
    let template = create_test_pattern(128, 128, 2);
    let target = create_test_pattern(128, 128, 2);

    // Run multiple algorithms to ensure no memory leaks
    let algorithms: Vec<Box<dyn AlignmentAlgorithm>> = vec![
        Box::new(OpenCVTemplateMatcher::new_ncc()),
        Box::new(OpenCVORB::new().with_fast_threshold(10)),
    ];

    for algorithm in algorithms {
        // Run multiple times - should not crash with memory issues
        for i in 0..5 {
            let result = algorithm.align(&template, &target);
            assert!(
                result.is_ok(),
                "Algorithm {} failed on iteration {}",
                algorithm.name(),
                i
            );

            let result = result.unwrap();
            assert!(result.processing_time_ms > 0.0);
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        }
    }
}

#[test]
fn test_parallel_safety() {
    use std::thread;

    let template = create_test_pattern(32, 32, 1);
    let target = create_test_pattern(32, 32, 1);

    // Test thread safety by running multiple algorithms in parallel
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let template = template.clone();
            let target = target.clone();

            thread::spawn(move || {
                let algorithm: Box<dyn AlignmentAlgorithm> = match i % 3 {
                    0 => Box::new(OpenCVTemplateMatcher::new_ncc()),
                    1 => Box::new(OpenCVTemplateMatcher::new_ssd()),
                    _ => Box::new(OpenCVORB::new().with_fast_threshold(10)),
                };

                let result = algorithm.align(&template, &target);
                (i, result)
            })
        })
        .collect();

    for handle in handles {
        let (thread_id, result) = handle.join().unwrap();
        assert!(result.is_ok(), "Thread {} failed", thread_id);

        let alignment_result = result.unwrap();
        assert!(alignment_result.processing_time_ms > 0.0);
        assert!(alignment_result.confidence >= 0.0 && alignment_result.confidence <= 1.0);
    }
}

#[test]
fn test_regression_detection() {
    // Baseline performance expectations
    let template = create_test_pattern(32, 32, 0);
    let target = create_test_pattern(32, 32, 0);

    struct PerformanceBaseline {
        algorithm: &'static str,
        max_time_ms: f32,
        min_confidence: f32,
        min_success_rate: f32,
    }

    let baselines = vec![
        PerformanceBaseline {
            algorithm: "OpenCV-NCC",
            max_time_ms: 200.0,
            min_confidence: 0.8,
            min_success_rate: 0.95,
        },
        PerformanceBaseline {
            algorithm: "OpenCV-SSD",
            max_time_ms: 200.0,
            min_confidence: 0.5,
            min_success_rate: 0.95,
        },
        PerformanceBaseline {
            algorithm: "OpenCV-ORB",
            max_time_ms: 1000.0,
            min_confidence: 0.0,
            min_success_rate: 0.7,
        },
    ];

    for baseline in baselines {
        let algorithm: Box<dyn AlignmentAlgorithm> = match baseline.algorithm {
            "OpenCV-NCC" => Box::new(OpenCVTemplateMatcher::new_ncc()),
            "OpenCV-SSD" => Box::new(OpenCVTemplateMatcher::new_ssd()),
            "OpenCV-ORB" => Box::new(OpenCVORB::new().with_fast_threshold(10)),
            _ => continue,
        };

        let (avg_time, avg_confidence, success_rate) =
            benchmark_algorithm(algorithm.as_ref(), &template, &target, 5);

        // Regression checks
        assert!(
            avg_time <= baseline.max_time_ms,
            "{} performance regression: {:.2}ms > {:.2}ms",
            baseline.algorithm,
            avg_time,
            baseline.max_time_ms
        );

        assert!(
            avg_confidence >= baseline.min_confidence,
            "{} confidence regression: {:.3} < {:.3}",
            baseline.algorithm,
            avg_confidence,
            baseline.min_confidence
        );

        assert!(
            success_rate >= baseline.min_success_rate,
            "{} success rate regression: {:.3} < {:.3}",
            baseline.algorithm,
            success_rate,
            baseline.min_success_rate
        );

        println!(
            "{} passed regression test: {:.2}ms, {:.3} confidence, {:.1}% success",
            baseline.algorithm,
            avg_time,
            avg_confidence,
            success_rate * 100.0
        );
    }
}
