use crate::pipeline::AlignmentResult;

pub fn print_results(results: &[AlignmentResult]) {
    println!("=== Alignment Results ===");
    for result in results {
        println!("Algorithm: {}", result.algorithm_name);

        if let Some(transform) = &result.transformation {
            println!(
                "  Translation: ({:.2}, {:.2})",
                transform.translation.0, transform.translation.1
            );
            println!("  Rotation: {:.2}°", transform.rotation_degrees);
            println!("  Scale: {:.2}", transform.scale);
        } else {
            println!("  Translation: (0.00, 0.00)");
            println!("  Rotation: 0.00°");
            println!("  Scale: 1.00");
        }

        println!("  Confidence: {:.2}", result.confidence);
        println!("  Processing Time: {:.2}ms", result.execution_time_ms);
        println!();
    }
}

pub fn print_comparison_table(results: &[AlignmentResult]) {
    println!("| Algorithm | Time (ms) | Translation | Rotation (°) | Scale | Confidence |");
    println!("|-----------|-----------|-------------|--------------|-------|------------|");

    for result in results {
        let (translation, rotation, scale) = if let Some(transform) = &result.transformation {
            (
                format!(
                    "({:.2}, {:.2})",
                    transform.translation.0, transform.translation.1
                ),
                format!("{:.2}", transform.rotation_degrees),
                format!("{:.2}", transform.scale),
            )
        } else {
            (
                "(0.00, 0.00)".to_string(),
                "0.00".to_string(),
                "1.00".to_string(),
            )
        };

        println!(
            "| {} | {:.2} | {} | {} | {} | {:.2} |",
            result.algorithm_name,
            result.execution_time_ms,
            translation,
            rotation,
            scale,
            result.confidence
        );
    }
}
