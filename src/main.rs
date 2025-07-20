use clap::{Parser, Subcommand};
use image_alignment::*;
use image_alignment::visualization::{print_results, print_comparison_table};
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

        /// Algorithm to use for alignment
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
        #[arg(short, long, default_value = "orb,phase")]
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

        /// Algorithms to benchmark
        #[arg(short, long, default_value = "all")]
        algorithms: String,

        /// Output file for benchmark results
        #[arg(short, long, default_value = "results/benchmark.json")]
        output: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

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
        Commands::Align { template, target, algorithm, output } => {
            handle_align(template, target, algorithm, output)?;
        }
        Commands::Compare { template, target, algorithms, output } => {
            handle_compare(template, target, algorithms, output)?;
        }
        Commands::Test { source, patch_size, count, output } => {
            handle_test(source, patch_size, count, output)?;
        }
        Commands::Benchmark { dataset, algorithms, output } => {
            handle_benchmark(dataset, algorithms, output)?;
        }
    }

    Ok(())
}

fn handle_align(template_path: PathBuf, target_path: PathBuf, algorithm: String, output: Option<PathBuf>) -> anyhow::Result<()> {
    println!("Loading images...");
    let template = load_image(&template_path)?;
    let target = load_image(&target_path)?;

    println!("Template: {}x{}, Target: {}x{}", 
             template.width(), template.height(), target.width(), target.height());

    // Validate image sizes
    validate_image_size(&template, 8)?;
    validate_image_size(&target, 16)?;

    println!("Running alignment with {} algorithm...", algorithm);
    
    let result = match algorithm.as_str() {
        "orb" => {
            let aligner = algorithms::OrbMatcher;
            aligner.align(&template, &target)?
        }
        "phase" => {
            let aligner = algorithms::PhaseCorrelation;
            aligner.align(&template, &target)?
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown algorithm: {}", algorithm));
        }
    };

    // Display results
    print_results(&[result.clone()]);

    // Save results if output specified
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(output_path, json)?;
        println!("Results saved to file.");
    }

    Ok(())
}

fn handle_compare(template_path: PathBuf, target_path: PathBuf, algorithms: String, output: Option<PathBuf>) -> anyhow::Result<()> {
    println!("Loading images...");
    let template = load_image(&template_path)?;
    let target = load_image(&target_path)?;

    println!("Template: {}x{}, Target: {}x{}", 
             template.width(), template.height(), target.width(), target.height());

    let algorithm_list: Vec<&str> = algorithms.split(',').map(|s| s.trim()).collect();
    let mut results = Vec::new();

    for algo in algorithm_list {
        println!("Running {} algorithm...", algo);
        let result = match algo {
            "orb" => {
                let aligner = algorithms::OrbMatcher;
                aligner.align(&template, &target)?
            }
            "phase" => {
                let aligner = algorithms::PhaseCorrelation;
                aligner.align(&template, &target)?
            }
            _ => {
                log::warn!("Unknown algorithm: {}, skipping", algo);
                continue;
            }
        };
        results.push(result);
    }

    // Display comparison
    print_comparison_table(&results);

    // Save results if output specified
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(output_path, json)?;
        println!("Comparison results saved to file.");
    }

    Ok(())
}


fn handle_test(source_path: PathBuf, patch_size: u32, count: u32, output_dir: PathBuf) -> anyhow::Result<()> {
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
    println!("Source image: {}x{}", source_image.width(), source_image.height());
    
    // Extract good patches
    println!("✂️ Extracting {} patches with good features...", count);
    let patches = PatchExtractor::extract_good_patches(&source_image, patch_size, count)?;
    println!("✅ Extracted {} patches", patches.len());
    
    // Save original patches
    let base_name = source_path.file_stem()
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
        ("rot_translate", GroundTruth { rotation_degrees: 20.0, translation_x: 8, translation_y: -5, scale_factor: 1.0 }),
    ];
    
    println!("🔄 Applying transformations and testing alignment...");
    let mut all_results = Vec::new();
    
    for (patch_idx, (patch, center_x, center_y)) in patches.iter().enumerate() {
        println!("\n--- Testing Patch {} (at {}x{}) ---", patch_idx + 1, center_x, center_y);
        
        for (transform_name, ground_truth) in &transformations {
            println!("  🔄 Applying transformation: {}", transform_name);
            
            // Apply transformation
            let transformed = if ground_truth.rotation_degrees != 0.0 && (ground_truth.translation_x != 0 || ground_truth.translation_y != 0) {
                ImageTransformer::rotate_and_translate(patch, ground_truth.rotation_degrees, ground_truth.translation_x, ground_truth.translation_y)?
            } else if ground_truth.rotation_degrees != 0.0 {
                ImageTransformer::rotate(patch, ground_truth.rotation_degrees)?
            } else {
                ImageTransformer::translate(patch, ground_truth.translation_x, ground_truth.translation_y)?
            };
            
            // Save transformed image
            let transformed_name = format!("{}_patch_{:02}_{}.png", base_name, patch_idx, transform_name);
            let transformed_path = transformed_dir.join(&transformed_name);
            transformed.save(&transformed_path)?;
            
            // Test both algorithms
            for algorithm in &["orb", "phase"] {
                println!("    🎯 Testing {} algorithm...", algorithm);
                
                let result = match *algorithm {
                    "orb" => {
                        let aligner = algorithms::OrbMatcher;
                        aligner.align(patch, &transformed)?
                    }
                    "phase" => {
                        let aligner = algorithms::PhaseCorrelation;
                        aligner.align(patch, &transformed)?
                    }
                    _ => continue,
                };
                
                // Calculate errors
                let rotation_error = (result.rotation - (-ground_truth.rotation_degrees)).abs();
                let translation_error_x = (result.translation.0 - (-ground_truth.translation_x as f32)).abs();
                let translation_error_y = (result.translation.1 - (-ground_truth.translation_y as f32)).abs();
                let translation_error_magnitude = (translation_error_x.powi(2) + translation_error_y.powi(2)).sqrt();
                
                println!("      📊 Results:");
                println!("         Detected: rotation={:.1}°, translation=({:.1}, {:.1})", 
                        result.rotation, result.translation.0, result.translation.1);
                println!("         Expected: rotation={:.1}°, translation=({}, {})", 
                        -ground_truth.rotation_degrees, -ground_truth.translation_x, -ground_truth.translation_y);
                println!("         Error: rotation={:.1}°, translation={:.1}px", 
                        rotation_error, translation_error_magnitude);
                println!("         Confidence: {:.2}, Time: {:.1}ms", result.confidence, result.processing_time_ms);
                
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
                        "rotation": result.rotation,
                        "translation": [result.translation.0, result.translation.1],
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

fn handle_benchmark(_dataset: PathBuf, _algorithms: String, _output: PathBuf) -> anyhow::Result<()> {
    println!("Benchmark functionality not yet implemented");
    Ok(())
}
#[cfg(test)]
mod tests {
    // No unit tests in main.rs - all tests are in tests/ directory
}
