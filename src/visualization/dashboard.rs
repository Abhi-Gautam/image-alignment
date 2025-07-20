use crate::AlignmentResult;

pub fn print_results(results: &[AlignmentResult]) {
    println!("=== Alignment Results ===");
    for result in results {
        println!("Algorithm: {}", result.algorithm_used);
        println!("  Translation: ({:.2}, {:.2})", result.translation.0, result.translation.1);
        println!("  Rotation: {:.2}°", result.rotation);
        println!("  Scale: {:.2}", result.scale);
        println!("  Confidence: {:.2}", result.confidence);
        println!("  Processing Time: {:.2}ms", result.processing_time_ms);
        println!();
    }
}

pub fn print_comparison_table(results: &[AlignmentResult]) {
    println!("| Algorithm | Time (ms) | Translation | Rotation (°) | Scale | Confidence |");
    println!("|-----------|-----------|-------------|--------------|-------|------------|");
    
    for result in results {
        println!("| {} | {:.2} | ({:.2}, {:.2}) | {:.2} | {:.2} | {:.2} |",
                 result.algorithm_used,
                 result.processing_time_ms,
                 result.translation.0, result.translation.1,
                 result.rotation,
                 result.scale,
                 result.confidence);
    }
}