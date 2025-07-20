use crate::{AlignmentResult, algorithms::AlignmentAlgorithm};
use image::GrayImage;
use instant::Instant;
use rayon::prelude::*;

/// Fast template matching algorithms for speed benchmarking
pub struct NormalizedCrossCorrelation;
pub struct SumOfSquaredDifferences;
pub struct IntegralImageMatcher;

impl AlignmentAlgorithm for NormalizedCrossCorrelation {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult> {
        let start = Instant::now();
        let result = ncc_template_match(template, target)?;
        
        let mut alignment_result = AlignmentResult::new("NCC");
        alignment_result.translation = result.translation;
        alignment_result.confidence = result.confidence;
        alignment_result.processing_time_ms = start.elapsed().as_millis() as f32;
        
        Ok(alignment_result)
    }

    fn name(&self) -> &'static str {
        "NCC"
    }
}

impl AlignmentAlgorithm for SumOfSquaredDifferences {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult> {
        let start = Instant::now();
        let result = ssd_template_match(template, target)?;
        
        let mut alignment_result = AlignmentResult::new("SSD");
        alignment_result.translation = result.translation;
        alignment_result.confidence = result.confidence;
        alignment_result.processing_time_ms = start.elapsed().as_millis() as f32;
        
        Ok(alignment_result)
    }

    fn name(&self) -> &'static str {
        "SSD"
    }
}

impl AlignmentAlgorithm for IntegralImageMatcher {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult> {
        let start = Instant::now();
        let result = integral_template_match(template, target)?;
        
        let mut alignment_result = AlignmentResult::new("IntegralImage");
        alignment_result.translation = result.translation;
        alignment_result.confidence = result.confidence;
        alignment_result.processing_time_ms = start.elapsed().as_millis() as f32;
        
        Ok(alignment_result)
    }

    fn name(&self) -> &'static str {
        "IntegralImage"
    }
}

struct MatchResult {
    translation: (f32, f32),
    confidence: f32,
}

/// Optimized Normalized Cross Correlation with sub-pixel accuracy
fn ncc_template_match(template: &GrayImage, target: &GrayImage) -> crate::Result<MatchResult> {
    let (t_width, t_height) = (template.width(), template.height());
    let (target_width, target_height) = (target.width(), target.height());
    
    if t_width > target_width || t_height > target_height {
        return Ok(MatchResult {
            translation: (0.0, 0.0),
            confidence: 0.0,
        });
    }

    // Pre-compute template statistics
    let template_mean = compute_mean(template);
    let template_std = compute_std(template, template_mean);
    
    if template_std < 1e-6 {
        return Ok(MatchResult {
            translation: (0.0, 0.0),
            confidence: 0.0,
        });
    }

    let search_width = target_width - t_width + 1;
    let search_height = target_height - t_height + 1;
    
    // Parallel search for best match
    let results: Vec<(f32, u32, u32)> = (0..search_height)
        .into_par_iter()
        .flat_map(|y| {
            (0..search_width).into_par_iter().map(move |x| {
                let score = calculate_ncc_fast(template, target, x, y, template_mean, template_std);
                (score, x, y)
            })
        })
        .collect();

    let (best_score, best_x, best_y) = results
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();

    // Sub-pixel refinement using parabolic interpolation
    let (sub_x, sub_y) = refine_peak_location(template, target, best_x, best_y, template_mean, template_std);

    Ok(MatchResult {
        translation: (sub_x, sub_y),
        confidence: (best_score + 1.0) / 2.0, // Normalize from [-1,1] to [0,1]
    })
}

/// Ultra-fast Sum of Squared Differences for pure translation
fn ssd_template_match(template: &GrayImage, target: &GrayImage) -> crate::Result<MatchResult> {
    let (t_width, t_height) = (template.width(), template.height());
    let (target_width, target_height) = (target.width(), target.height());
    
    if t_width > target_width || t_height > target_height {
        return Ok(MatchResult {
            translation: (0.0, 0.0),
            confidence: 0.0,
        });
    }

    let search_width = target_width - t_width + 1;
    let search_height = target_height - t_height + 1;
    
    // Parallel SSD computation
    let results: Vec<(f32, u32, u32)> = (0..search_height)
        .into_par_iter()
        .flat_map(|y| {
            (0..search_width).into_par_iter().map(move |x| {
                let ssd = calculate_ssd(template, target, x, y);
                (ssd, x, y)
            })
        })
        .collect();

    let (best_ssd, best_x, best_y) = results
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();

    // Convert SSD to confidence (lower SSD = higher confidence)
    let max_possible_ssd = (t_width * t_height) as f32 * 255.0 * 255.0;
    let confidence = 1.0 - (best_ssd / max_possible_ssd).min(1.0);

    Ok(MatchResult {
        translation: (best_x as f32, best_y as f32),
        confidence,
    })
}

/// Integral image-based template matching (10-20x speedup for normalization)
fn integral_template_match(template: &GrayImage, target: &GrayImage) -> crate::Result<MatchResult> {
    let (t_width, t_height) = (template.width(), template.height());
    let (target_width, target_height) = (target.width(), target.height());
    
    if t_width > target_width || t_height > target_height {
        return Ok(MatchResult {
            translation: (0.0, 0.0),
            confidence: 0.0,
        });
    }

    // Build integral images for fast mean computation
    let integral_img = build_integral_image(target);
    let integral_img_sq = build_integral_image_squared(target);
    
    let template_mean = compute_mean(template);
    let template_std = compute_std(template, template_mean);
    
    if template_std < 1e-6 {
        return Ok(MatchResult {
            translation: (0.0, 0.0),
            confidence: 0.0,
        });
    }

    let search_width = target_width - t_width + 1;
    let search_height = target_height - t_height + 1;
    
    let mut best_score = f32::NEG_INFINITY;
    let mut best_x = 0;
    let mut best_y = 0;

    for y in 0..search_height {
        for x in 0..search_width {
            // Fast mean and std computation using integral images
            let region_mean = compute_region_mean(&integral_img, x, y, t_width, t_height);
            let region_std = compute_region_std(&integral_img_sq, &integral_img, x, y, t_width, t_height, region_mean);
            
            if region_std < 1e-6 {
                continue;
            }

            let score = calculate_ncc_with_precomputed_stats(
                template, target, x, y, 
                template_mean, template_std, 
                region_mean, region_std
            );
            
            if score > best_score {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    Ok(MatchResult {
        translation: (best_x as f32, best_y as f32),
        confidence: (best_score + 1.0) / 2.0,
    })
}

// Helper functions for optimized computations

fn compute_mean(image: &GrayImage) -> f32 {
    let sum: u64 = image.pixels().map(|p| p[0] as u64).sum();
    sum as f32 / (image.width() * image.height()) as f32
}

fn compute_std(image: &GrayImage, mean: f32) -> f32 {
    let variance: f32 = image.pixels()
        .map(|p| {
            let diff = p[0] as f32 - mean;
            diff * diff
        })
        .sum::<f32>() / (image.width() * image.height()) as f32;
    variance.sqrt()
}

fn calculate_ncc_fast(
    template: &GrayImage, 
    target: &GrayImage, 
    offset_x: u32, 
    offset_y: u32,
    template_mean: f32,
    template_std: f32,
) -> f32 {
    let (t_width, t_height) = (template.width(), template.height());
    
    // Compute target region statistics
    let mut sum_target = 0.0;
    let mut sum_target_sq = 0.0;
    let mut sum_product = 0.0;
    
    for y in 0..t_height {
        for x in 0..t_width {
            let template_val = template.get_pixel(x, y)[0] as f32;
            let target_val = target.get_pixel(x + offset_x, y + offset_y)[0] as f32;
            
            sum_target += target_val;
            sum_target_sq += target_val * target_val;
            sum_product += template_val * target_val;
        }
    }
    
    let n = (t_width * t_height) as f32;
    let target_mean = sum_target / n;
    let target_variance = (sum_target_sq / n) - (target_mean * target_mean);
    let target_std = target_variance.sqrt();
    
    if target_std < 1e-6 {
        return 0.0;
    }
    
    let numerator = (sum_product / n) - (template_mean * target_mean);
    numerator / (template_std * target_std)
}

fn calculate_ssd(template: &GrayImage, target: &GrayImage, offset_x: u32, offset_y: u32) -> f32 {
    let (t_width, t_height) = (template.width(), template.height());
    let mut ssd = 0.0;
    
    for y in 0..t_height {
        for x in 0..t_width {
            let template_val = template.get_pixel(x, y)[0] as f32;
            let target_val = target.get_pixel(x + offset_x, y + offset_y)[0] as f32;
            let diff = template_val - target_val;
            ssd += diff * diff;
        }
    }
    
    ssd
}

fn build_integral_image(image: &GrayImage) -> Vec<Vec<u64>> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut integral = vec![vec![0u64; width + 1]; height + 1];
    
    for y in 1..=height {
        for x in 1..=width {
            let pixel_val = image.get_pixel((x - 1) as u32, (y - 1) as u32)[0] as u64;
            integral[y][x] = pixel_val + integral[y-1][x] + integral[y][x-1] - integral[y-1][x-1];
        }
    }
    
    integral
}

fn build_integral_image_squared(image: &GrayImage) -> Vec<Vec<u64>> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut integral = vec![vec![0u64; width + 1]; height + 1];
    
    for y in 1..=height {
        for x in 1..=width {
            let pixel_val = image.get_pixel((x - 1) as u32, (y - 1) as u32)[0] as u64;
            let pixel_sq = pixel_val * pixel_val;
            integral[y][x] = pixel_sq + integral[y-1][x] + integral[y][x-1] - integral[y-1][x-1];
        }
    }
    
    integral
}

fn compute_region_mean(integral: &[Vec<u64>], x: u32, y: u32, width: u32, height: u32) -> f32 {
    let (x, y, x2, y2) = (x as usize, y as usize, (x + width) as usize, (y + height) as usize);
    let sum = integral[y2][x2] + integral[y][x] - integral[y][x2] - integral[y2][x];
    sum as f32 / (width * height) as f32
}

fn compute_region_std(
    integral_sq: &[Vec<u64>], 
    _integral: &[Vec<u64>], 
    x: u32, 
    y: u32, 
    width: u32, 
    height: u32,
    mean: f32,
) -> f32 {
    let (x, y, x2, y2) = (x as usize, y as usize, (x + width) as usize, (y + height) as usize);
    let sum_sq = integral_sq[y2][x2] + integral_sq[y][x] - integral_sq[y][x2] - integral_sq[y2][x];
    let n = (width * height) as f32;
    let variance = (sum_sq as f32 / n) - (mean * mean);
    variance.max(0.0).sqrt()
}

fn calculate_ncc_with_precomputed_stats(
    template: &GrayImage,
    target: &GrayImage,
    offset_x: u32,
    offset_y: u32,
    template_mean: f32,
    template_std: f32,
    target_mean: f32,
    target_std: f32,
) -> f32 {
    let (t_width, t_height) = (template.width(), template.height());
    let mut sum_product = 0.0;
    
    for y in 0..t_height {
        for x in 0..t_width {
            let template_val = template.get_pixel(x, y)[0] as f32;
            let target_val = target.get_pixel(x + offset_x, y + offset_y)[0] as f32;
            sum_product += template_val * target_val;
        }
    }
    
    let n = (t_width * t_height) as f32;
    let numerator = (sum_product / n) - (template_mean * target_mean);
    numerator / (template_std * target_std)
}

fn refine_peak_location(
    template: &GrayImage,
    target: &GrayImage,
    peak_x: u32,
    peak_y: u32,
    template_mean: f32,
    template_std: f32,
) -> (f32, f32) {
    // Simple sub-pixel refinement using neighboring scores
    let (target_width, target_height) = (target.width(), target.height());
    let (t_width, t_height) = (template.width(), template.height());
    
    if peak_x == 0 || peak_y == 0 || 
       peak_x >= target_width - t_width || peak_y >= target_height - t_height {
        return (peak_x as f32, peak_y as f32);
    }
    
    // Get neighboring correlation scores
    let score_center = calculate_ncc_fast(template, target, peak_x, peak_y, template_mean, template_std);
    let score_left = calculate_ncc_fast(template, target, peak_x - 1, peak_y, template_mean, template_std);
    let score_right = calculate_ncc_fast(template, target, peak_x + 1, peak_y, template_mean, template_std);
    let score_up = calculate_ncc_fast(template, target, peak_x, peak_y - 1, template_mean, template_std);
    let score_down = calculate_ncc_fast(template, target, peak_x, peak_y + 1, template_mean, template_std);
    
    // Parabolic interpolation for sub-pixel accuracy
    let dx = if score_left != score_right {
        0.5 * (score_left - score_right) / (score_left - 2.0 * score_center + score_right)
    } else {
        0.0
    };
    
    let dy = if score_up != score_down {
        0.5 * (score_up - score_down) / (score_up - 2.0 * score_center + score_down)
    } else {
        0.0
    };
    
    (peak_x as f32 + dx, peak_y as f32 + dy)
}