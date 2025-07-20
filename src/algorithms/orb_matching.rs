use crate::{AlignmentResult, algorithms::AlignmentAlgorithm};
use image::GrayImage;
use instant::Instant;

pub struct OrbMatcher;

impl AlignmentAlgorithm for OrbMatcher {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult> {
        let start = Instant::now();
        
        // Simple template matching as a placeholder for ORB
        let result = template_match(template, target)?;
        
        let mut alignment_result = AlignmentResult::new("ORB");
        alignment_result.translation = result.translation;
        alignment_result.confidence = result.confidence;
        alignment_result.processing_time_ms = start.elapsed().as_millis() as f32;
        
        Ok(alignment_result)
    }

    fn name(&self) -> &'static str {
        "ORB"
    }
}

struct MatchResult {
    translation: (f32, f32),
    confidence: f32,
}

fn template_match(template: &GrayImage, target: &GrayImage) -> crate::Result<MatchResult> {
    let (t_width, t_height) = (template.width(), template.height());
    let (target_width, target_height) = (target.width(), target.height());
    
    if t_width > target_width || t_height > target_height {
        return Err(anyhow::anyhow!("Template larger than target image"));
    }
    
    let mut best_score = f32::NEG_INFINITY;
    let mut best_x = 0;
    let mut best_y = 0;
    
    // Search for best match position
    for y in 0..=(target_height - t_height) {
        for x in 0..=(target_width - t_width) {
            let score = calculate_ncc(template, target, x, y);
            if score > best_score {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }
    
    Ok(MatchResult {
        translation: (best_x as f32, best_y as f32),
        confidence: (best_score + 1.0) / 2.0, // Normalize NCC from [-1,1] to [0,1]
    })
}

// Normalized Cross Correlation
fn calculate_ncc(template: &GrayImage, target: &GrayImage, offset_x: u32, offset_y: u32) -> f32 {
    let (t_width, t_height) = (template.width(), template.height());
    
    let mut sum_template = 0.0;
    let mut sum_target = 0.0;
    let mut sum_template_sq = 0.0;
    let mut sum_target_sq = 0.0;
    let mut sum_product = 0.0;
    let n = (t_width * t_height) as f32;
    
    // Calculate means and correlations
    for y in 0..t_height {
        for x in 0..t_width {
            let template_val = template.get_pixel(x, y)[0] as f32;
            let target_val = target.get_pixel(x + offset_x, y + offset_y)[0] as f32;
            
            sum_template += template_val;
            sum_target += target_val;
            sum_template_sq += template_val * template_val;
            sum_target_sq += target_val * target_val;
            sum_product += template_val * target_val;
        }
    }
    
    let mean_template = sum_template / n;
    let mean_target = sum_target / n;
    
    let numerator = sum_product - n * mean_template * mean_target;
    let denominator = ((sum_template_sq - n * mean_template * mean_template) * 
                      (sum_target_sq - n * mean_target * mean_target)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}