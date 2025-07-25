use clap::{Parser, Subcommand};
use image_alignment::config::load_config_or_default;
use image_alignment::pipeline::AlignmentAlgorithm;
use image_alignment::utils::{grayimage_to_mat, load_image, validate_image_size_with_limits};
use image_alignment::*;
use opencv::core::Mat;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "align")]
#[command(about = "Real-time unsupervised semiconductor wafer image alignment system")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Path to configuration file (TOML or JSON)
    #[arg(short = 'c', long)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Align a template image to a target image
    Align {
        /// Path to the template image (patch)
        #[arg(short, long)]
        template: PathBuf,

        /// Path to the target image (SEM image)
        #[arg(short = 'T', long)]
        target: PathBuf,

        /// Algorithm to use for alignment (orb, ncc, ssd, ccorr, akaze, sift)
        #[arg(short, long, default_value = "orb")]
        algorithm: String,

        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compare multiple algorithms on the same image pair
    Compare {
        /// Path to the template image (patch)
        #[arg(short, long)]
        template: PathBuf,

        /// Path to the target image (SEM image)
        #[arg(short = 'T', long)]
        target: PathBuf,

        /// Algorithms to compare (comma-separated)
        #[arg(short, long, default_value = "orb,ncc,ssd,akaze")]
        algorithms: String,

        /// Output file for comparison results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Test alignment accuracy with known transformations
    Test {
        /// Path to source SEM image
        #[arg(short, long)]
        source: PathBuf,

        /// Patch size to extract (32, 64, or 128)
        #[arg(short, long, default_value = "64")]
        patch_size: u32,

        /// Number of test patches to extract
        #[arg(short = 'n', long, default_value = "3")]
        count: u32,

        /// Output directory for test results
        #[arg(short, long, default_value = "results/validation")]
        output: PathBuf,
    },

    /// Run benchmarks on a dataset
    Benchmark {
        /// Path to dataset directory
        #[arg(short, long)]
        dataset: PathBuf,

        /// Algorithms to benchmark (orb, ncc, ssd, ccorr, akaze, sift, all)
        #[arg(short, long, default_value = "all")]
        algorithms: String,

        /// Output file for benchmark results
        #[arg(short, long, default_value = "results/benchmark.json")]
        output: PathBuf,
    },

    /// Run comprehensive visual tests with detailed reports
    VisualTest {
        /// Path to SEM image for testing
        #[arg(short, long)]
        sem_image: PathBuf,

        /// Output directory for visual test results
        #[arg(short, long, default_value = "results/visual_tests")]
        output: PathBuf,

        /// Patch sizes to test (comma-separated, e.g., "32,64,128")
        #[arg(long, default_value = "32,64,128")]
        patch_sizes: String,

        /// Test scenarios to run (comma-separated: clean,translation,rotation,noise,blur,brightness,scale,complex)
        #[arg(
            long,
            default_value = "clean,translation_5px,translation_10px,rotation_10deg,rotation_30deg,gaussian_noise,salt_pepper,gaussian_blur,brightness_change,scale_120"
        )]
        scenarios: String,
    },

    /// Launch web dashboard to visualize test results
    Dashboard {
        /// Directory containing test results
        #[arg(short, long, default_value = "results")]
        results_dir: PathBuf,

        /// Port to serve dashboard on
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load configuration from file or use default
    let _config = load_config_or_default(cli.config.as_ref().map(|p| p.to_str().unwrap()));

    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(match cli.verbose {
            0 => log::LevelFilter::Warn,
            1 => log::LevelFilter::Info,
            2 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        })
        .init();

    match cli.command {
        Commands::Align {
            template,
            target,
            algorithm,
            output,
        } => {
            handle_align(template, target, algorithm, output)?;
        }
        Commands::Compare {
            template,
            target,
            algorithms,
            output,
        } => {
            handle_compare(template, target, algorithms, output)?;
        }
        Commands::Test {
            source,
            patch_size,
            count,
            output,
        } => {
            handle_test(source, patch_size, count, output)?;
        }
        Commands::Benchmark {
            dataset,
            algorithms,
            output,
        } => {
            handle_benchmark(dataset, algorithms, output)?;
        }
        Commands::VisualTest {
            sem_image,
            output,
            patch_sizes,
            scenarios,
        } => {
            handle_visual_test(sem_image, output, patch_sizes, scenarios)?;
        }
        Commands::Dashboard { results_dir, port } => {
            handle_dashboard(results_dir, port).await?;
        }
    }

    Ok(())
}

fn convert_pipeline_to_main_result(pipeline_result: pipeline::AlignmentResult) -> AlignmentResult {
    let (translation, rotation, scale) = if let Some(transform) = &pipeline_result.transformation {
        (transform.translation, transform.rotation_degrees, transform.scale)
    } else {
        // Default values if no transformation is available
        ((0.0, 0.0), 0.0, 1.0)
    };

    AlignmentResult {
        translation,
        rotation,
        scale,
        confidence: pipeline_result.confidence as f32,
        processing_time_ms: pipeline_result.execution_time_ms as f32,
        algorithm_used: pipeline_result.algorithm_name,
    }
}

fn create_algorithm_and_align(
    algorithm: &str,
    target_mat: &Mat,
    template_mat: &Mat,
) -> anyhow::Result<AlignmentResult> {
    let pipeline_result = match algorithm {
        "orb" | "opencv-orb" => {
            let aligner = algorithms::OpenCVORB::new()?;
            aligner.align(target_mat, template_mat)?
        }
        "ncc" | "opencv-ncc" => {
            let aligner = algorithms::OpenCVTemplateMatcher::new_ncc();
            aligner.align(target_mat, template_mat)?
        }
        "ssd" | "opencv-ssd" => {
            let aligner = algorithms::OpenCVTemplateMatcher::new_ssd();
            aligner.align(target_mat, template_mat)?
        }
        "ccorr" | "opencv-ccorr" => {
            let aligner = algorithms::OpenCVTemplateMatcher::new_ccorr();
            aligner.align(target_mat, template_mat)?
        }
        "akaze" | "opencv-akaze" => {
            let aligner = algorithms::OpenCVAKAZE::new()?;
            aligner.align(target_mat, template_mat)?
        }
        "sift" | "opencv-sift" => {
            let aligner = algorithms::OpenCVSIFT::new()?;
            aligner.align(target_mat, template_mat)?
        }
        _ => return Err(anyhow::anyhow!(
            "Unknown algorithm: {}. Available: orb, ncc, ssd, ccorr, akaze, sift",
            algorithm
        )),
    };
    
    Ok(convert_pipeline_to_main_result(pipeline_result))
}

fn print_main_results(results: &[AlignmentResult]) {
    println!("\n📊 Alignment Results:");
    println!("┌─────────────────────┬─────────────────────┬──────────────┬─────────────────────┬─────────────────────┐");
    println!("│ Algorithm           │ Translation (x, y)  │ Rotation     │ Scale               │ Confidence          │");
    println!("├─────────────────────┼─────────────────────┼──────────────┼─────────────────────┼─────────────────────┤");
    
    for result in results {
        println!(
            "│ {:<19} │ ({:>6.1}, {:>6.1})   │ {:>10.1}°   │ {:>17.3}x   │ {:>17.3}     │",
            result.algorithm_used,
            result.translation.0,
            result.translation.1,
            result.rotation,
            result.scale,
            result.confidence
        );
    }
    
    println!("└─────────────────────┴─────────────────────┴──────────────┴─────────────────────┴─────────────────────┘");
    
    for result in results {
        println!(
            "⏱️  {}: {:.1}ms",
            result.algorithm_used, result.processing_time_ms
        );
    }
}

fn handle_align(
    template_path: PathBuf,
    target_path: PathBuf,
    algorithm: String,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    println!("Loading images...");
    let template = load_image(&template_path)?;
    let target = load_image(&target_path)?;

    println!(
        "Template: {}x{}, Target: {}x{}",
        template.width(),
        template.height(),
        target.width(),
        target.height()
    );

    // Validate image sizes
    validate_image_size_with_limits(&template, 8, 10000)?;
    validate_image_size_with_limits(&target, 16, 10000)?;

    // Convert GrayImage to Mat
    let template_mat = grayimage_to_mat(&template)?;
    let target_mat = grayimage_to_mat(&target)?;

    println!("Running alignment with {} algorithm...", algorithm);

    let result = create_algorithm_and_align(&algorithm, &target_mat, &template_mat)?;

    // Display results
    print_main_results(&[result.clone()]);

    // Save results if output specified
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(output_path, json)?;
        println!("Results saved to file.");
    }

    Ok(())
}

fn handle_compare(
    template_path: PathBuf,
    target_path: PathBuf,
    algorithms: String,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    println!("Loading images...");
    let template = load_image(&template_path)?;
    let target = load_image(&target_path)?;

    println!(
        "Template: {}x{}, Target: {}x{}",
        template.width(),
        template.height(),
        target.width(),
        target.height()
    );

    // Convert GrayImage to Mat
    let template_mat = grayimage_to_mat(&template)?;
    let target_mat = grayimage_to_mat(&target)?;

    let algorithm_list: Vec<&str> = algorithms.split(',').map(|s| s.trim()).collect();
    let mut results = Vec::new();

    for algo in algorithm_list {
        println!("Running {} algorithm...", algo);
        match create_algorithm_and_align(algo, &target_mat, &template_mat) {
            Ok(result) => results.push(result),
            Err(_) => {
                log::warn!("Unknown algorithm: {}, skipping. Available: orb, ncc, ssd, ccorr, akaze, sift", algo);
                continue;
            }
        }
    }

    // Display comparison
    print_main_results(&results);

    // Save results if output specified
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(output_path, json)?;
        println!("Comparison results saved to file.");
    }

    Ok(())
}

fn handle_test(
    source_path: PathBuf,
    patch_size: u32,
    count: u32,
    output_dir: PathBuf,
) -> anyhow::Result<()> {
    println!("🧪 Starting alignment accuracy testing...");
    println!("Source: {:?}", source_path);
    println!("Patch size: {}x{}", patch_size, patch_size);
    println!("Test patches: {}", count);

    // Create output directories
    std::fs::create_dir_all(&output_dir)?;
    let patches_dir = output_dir.join("patches");
    let transformed_dir = output_dir.join("transformed");
    let results_dir = output_dir.join("results");
    std::fs::create_dir_all(&patches_dir)?;
    std::fs::create_dir_all(&transformed_dir)?;
    std::fs::create_dir_all(&results_dir)?;

    // Load source image
    println!("📸 Loading source SEM image...");
    let source_image = load_image(&source_path)?;
    println!(
        "Source image: {}x{}",
        source_image.width(),
        source_image.height()
    );

    // Extract good patches
    println!("✂️ Extracting {} patches with good features...", count);
    let patches = PatchExtractor::extract_good_patches(&source_image, patch_size, count)?;
    println!("✅ Extracted {} patches", patches.len());

    // Save original patches
    let base_name = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("wafer");
    let _patch_paths = PatchExtractor::save_patches(&patches, base_name, &patches_dir)?;

    // Define test transformations
    let transformations = vec![
        ("rot_15deg", GroundTruth::rotation(15.0)),
        ("rot_30deg", GroundTruth::rotation(30.0)),
        ("rot_45deg", GroundTruth::rotation(45.0)),
        ("translate_5px", GroundTruth::translation(5, 3)),
        ("translate_10px", GroundTruth::translation(10, -7)),
        (
            "rot_translate",
            GroundTruth {
                rotation_degrees: 20.0,
                translation_x: 8,
                translation_y: -5,
                scale_factor: 1.0,
            },
        ),
    ];

    println!("🔄 Applying transformations and testing alignment...");
    let mut all_results = Vec::new();

    for (patch_idx, (patch, center_x, center_y)) in patches.iter().enumerate() {
        println!(
            "\n--- Testing Patch {} (at {}x{}) ---",
            patch_idx + 1,
            center_x,
            center_y
        );

        for (transform_name, ground_truth) in &transformations {
            println!("  🔄 Applying transformation: {}", transform_name);

            // Apply transformation
            let transformed = if ground_truth.rotation_degrees != 0.0
                && (ground_truth.translation_x != 0 || ground_truth.translation_y != 0)
            {
                ImageTransformer::rotate_and_translate(
                    patch,
                    ground_truth.rotation_degrees,
                    ground_truth.translation_x,
                    ground_truth.translation_y,
                )?
            } else if ground_truth.rotation_degrees != 0.0 {
                ImageTransformer::rotate(patch, ground_truth.rotation_degrees)?
            } else {
                ImageTransformer::translate(
                    patch,
                    ground_truth.translation_x,
                    ground_truth.translation_y,
                )?
            };

            // Save transformed image
            let transformed_name = format!(
                "{}_patch_{:02}_{}.png",
                base_name, patch_idx, transform_name
            );
            let transformed_path = transformed_dir.join(&transformed_name);
            transformed.save(&transformed_path)?;

            // Convert images to Mat
            let patch_mat = grayimage_to_mat(patch)?;
            let transformed_mat = grayimage_to_mat(&transformed)?;

            // Test OpenCV algorithms
            for algorithm in &["orb", "ncc", "ssd", "akaze", "sift"] {
                println!("    🎯 Testing {} algorithm...", algorithm);

                let result = match create_algorithm_and_align(algorithm, &transformed_mat, &patch_mat) {
                    Ok(result) => result,
                    Err(_) => continue,
                };

                // Calculate errors (extract rotation and translation)
                let detected_rotation = result.rotation;
                let detected_translation = result.translation;

                let rotation_error = (detected_rotation - (-ground_truth.rotation_degrees)).abs();
                let translation_error_x =
                    (detected_translation.0 - (-ground_truth.translation_x as f32)).abs();
                let translation_error_y =
                    (detected_translation.1 - (-ground_truth.translation_y as f32)).abs();
                let translation_error_magnitude =
                    (translation_error_x.powi(2) + translation_error_y.powi(2)).sqrt();

                println!("      📊 Results:");
                println!(
                    "         Detected: rotation={:.1}°, translation=({:.1}, {:.1})",
                    detected_rotation, detected_translation.0, detected_translation.1
                );
                println!(
                    "         Expected: rotation={:.1}°, translation=({}, {})",
                    -ground_truth.rotation_degrees,
                    -ground_truth.translation_x,
                    -ground_truth.translation_y
                );
                println!(
                    "         Error: rotation={:.1}°, translation={:.1}px",
                    rotation_error, translation_error_magnitude
                );
                println!(
                    "         Confidence: {:.2}, Time: {:.1}ms",
                    result.confidence, result.processing_time_ms
                );

                // Store result with metadata
                let test_result = serde_json::json!({
                    "patch_index": patch_idx,
                    "patch_center": [center_x, center_y],
                    "transformation": transform_name,
                    "algorithm": algorithm,
                    "ground_truth": {
                        "rotation": ground_truth.rotation_degrees,
                        "translation": [ground_truth.translation_x, ground_truth.translation_y]
                    },
                    "detected": {
                        "rotation": detected_rotation,
                        "translation": [detected_translation.0, detected_translation.1],
                        "confidence": result.confidence,
                        "processing_time_ms": result.processing_time_ms
                    },
                    "errors": {
                        "rotation_degrees": rotation_error,
                        "translation_x_pixels": translation_error_x,
                        "translation_y_pixels": translation_error_y,
                        "translation_magnitude_pixels": translation_error_magnitude
                    }
                });

                all_results.push(test_result);
            }
        }
    }

    // Save comprehensive results
    let results_file = results_dir.join("accuracy_validation.json");
    let results_json = serde_json::to_string_pretty(&all_results)?;
    std::fs::write(&results_file, results_json)?;

    // Generate summary
    println!("\n📊 === VALIDATION SUMMARY ===");
    println!("Total tests: {}", all_results.len());
    println!("Results saved to: {:?}", results_file);
    println!("\nDetailed results available in:");
    println!("  Patches: {:?}", patches_dir);
    println!("  Transformed: {:?}", transformed_dir);
    println!("  Results: {:?}", results_dir);

    Ok(())
}

fn handle_benchmark(
    _dataset: PathBuf,
    _algorithms: String,
    _output: PathBuf,
) -> anyhow::Result<()> {
    println!("Benchmark functionality not yet implemented");
    Ok(())
}

fn handle_visual_test(
    sem_image_path: PathBuf,
    output_dir: PathBuf,
    patch_sizes: String,
    scenarios: String,
) -> anyhow::Result<()> {
    use image_alignment::visualization::VisualTester;

    // Parse patch sizes
    let patch_sizes: Vec<u32> = patch_sizes
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Invalid patch size: {}", e))?;

    // Parse scenarios
    let scenario_filters: Vec<String> =
        scenarios.split(',').map(|s| s.trim().to_string()).collect();

    println!("🔬 Starting comprehensive visual testing...");
    println!("📁 SEM Image: {}", sem_image_path.display());
    println!("📂 Output Directory: {}", output_dir.display());
    println!("🎯 Patch Sizes: {:?}", patch_sizes);
    println!("🎯 Scenarios: {:?}", scenario_filters);

    let mut tester = VisualTester::new(output_dir);
    let reports = tester.run_comprehensive_test(&sem_image_path, Some(&patch_sizes), Some(&scenario_filters))?;

    println!("\n✅ Comprehensive visual testing completed!");
    println!("📊 Generated {} test reports", reports.len());
    println!("📋 Detailed analysis: COMPREHENSIVE_ANALYSIS_REPORT.md");

    // Print quick summary
    let mut by_algorithm = std::collections::HashMap::new();
    for report in &reports {
        by_algorithm
            .entry(&report.algorithm_name)
            .or_insert_with(Vec::new)
            .push(report);
    }

    println!("\n🎯 Quick Performance Summary:");
    println!("┌─────────────────────┬─────────┬──────────────┬─────────────────────┐");
    println!("│ Algorithm           │ Tests   │ Success Rate │ Avg Time (ms)       │");
    println!("├─────────────────────┼─────────┼──────────────┼─────────────────────┤");

    for (algo_name, algo_reports) in by_algorithm {
        let success_count = algo_reports
            .iter()
            .filter(|r| r.performance_metrics.success)
            .count();
        let success_rate = success_count as f32 / algo_reports.len() as f32 * 100.0;
        let avg_time: f32 = algo_reports
            .iter()
            .map(|r| r.performance_metrics.processing_time_ms)
            .sum::<f32>()
            / algo_reports.len() as f32;

        println!(
            "│ {:<19} │ {:<7} │ {:<12.1}% │ {:<19.1} │",
            algo_name,
            algo_reports.len(),
            success_rate,
            avg_time
        );
    }

    println!("└─────────────────────┴─────────┴──────────────┴─────────────────────┘");

    Ok(())
}

async fn handle_dashboard(results_dir: PathBuf, port: u16) -> anyhow::Result<()> {
    use image_alignment::dashboard::start_dashboard_server;

    println!("🚀 Starting SEM Image Alignment Dashboard...");
    println!("📁 Results Directory: {}", results_dir.display());
    println!("🌐 Port: {}", port);
    println!();

    start_dashboard_server(results_dir, port).await
}

