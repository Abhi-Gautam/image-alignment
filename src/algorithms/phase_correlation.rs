use crate::{AlignmentResult, algorithms::AlignmentAlgorithm};
use image::GrayImage;
use instant::Instant;
use rustfft::{FftPlanner, num_complex::Complex};
use ndarray::Array2;

pub struct PhaseCorrelation;

impl AlignmentAlgorithm for PhaseCorrelation {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult> {
        let start = Instant::now();
        
        // For now, implement a simple version that works on same-sized images
        let (tx, ty) = phase_correlate(template, target)?;
        
        let mut result = AlignmentResult::new("PhaseCorrelation");
        result.translation = (tx, ty);
        result.confidence = if tx == 0.0 && ty == 0.0 { 1.0 } else { 0.8 }; // Placeholder confidence
        result.processing_time_ms = start.elapsed().as_millis() as f32;
        
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "PhaseCorrelation"
    }
}

fn phase_correlate(template: &GrayImage, target: &GrayImage) -> crate::Result<(f32, f32)> {
    let (t_width, t_height) = (template.width() as usize, template.height() as usize);
    let (target_width, target_height) = (target.width() as usize, target.height() as usize);
    
    // For simplicity, only handle same-sized images for now
    if t_width != target_width || t_height != target_height {
        return Ok((0.0, 0.0)); // No translation found
    }
    
    // Convert images to complex arrays
    let template_complex = image_to_complex(template);
    let target_complex = image_to_complex(target);
    
    // Compute FFTs
    let template_fft = compute_2d_fft(&template_complex)?;
    let target_fft = compute_2d_fft(&target_complex)?;
    
    // Compute cross-power spectrum
    let cross_power = compute_cross_power_spectrum(&template_fft, &target_fft);
    
    // Compute inverse FFT
    let correlation = compute_2d_ifft(&cross_power)?;
    
    // Find peak in correlation
    let (peak_x, peak_y) = find_correlation_peak(&correlation);
    
    // Convert to translation (handle wraparound)
    let tx = if peak_x > t_width / 2 { 
        peak_x as f32 - t_width as f32 
    } else { 
        peak_x as f32 
    };
    let ty = if peak_y > t_height / 2 { 
        peak_y as f32 - t_height as f32 
    } else { 
        peak_y as f32 
    };
    
    Ok((tx, ty))
}

fn image_to_complex(img: &GrayImage) -> Array2<Complex<f32>> {
    let (width, height) = (img.width() as usize, img.height() as usize);
    let mut result = Array2::zeros((height, width));
    
    for y in 0..height {
        for x in 0..width {
            let pixel_value = img.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
            result[[y, x]] = Complex::new(pixel_value, 0.0);
        }
    }
    
    result
}

fn compute_2d_fft(input: &Array2<Complex<f32>>) -> crate::Result<Array2<Complex<f32>>> {
    let (height, width) = input.dim();
    let mut result = input.clone();
    
    let mut planner = FftPlanner::new();
    
    // FFT along rows
    let fft_row = planner.plan_fft_forward(width);
    for mut row in result.rows_mut() {
        let mut row_data: Vec<Complex<f32>> = row.to_vec();
        fft_row.process(&mut row_data);
        for (i, val) in row_data.iter().enumerate() {
            row[i] = *val;
        }
    }
    
    // FFT along columns
    let fft_col = planner.plan_fft_forward(height);
    for mut col in result.columns_mut() {
        let mut col_data: Vec<Complex<f32>> = col.to_vec();
        fft_col.process(&mut col_data);
        for (i, val) in col_data.iter().enumerate() {
            col[i] = *val;
        }
    }
    
    Ok(result)
}

fn compute_2d_ifft(input: &Array2<Complex<f32>>) -> crate::Result<Array2<Complex<f32>>> {
    let (height, width) = input.dim();
    let mut result = input.clone();
    
    let mut planner = FftPlanner::new();
    
    // IFFT along rows
    let ifft_row = planner.plan_fft_inverse(width);
    for mut row in result.rows_mut() {
        let mut row_data: Vec<Complex<f32>> = row.to_vec();
        ifft_row.process(&mut row_data);
        for (i, val) in row_data.iter().enumerate() {
            row[i] = *val / width as f32;
        }
    }
    
    // IFFT along columns
    let ifft_col = planner.plan_fft_inverse(height);
    for mut col in result.columns_mut() {
        let mut col_data: Vec<Complex<f32>> = col.to_vec();
        ifft_col.process(&mut col_data);
        for (i, val) in col_data.iter().enumerate() {
            col[i] = *val / height as f32;
        }
    }
    
    Ok(result)
}

fn compute_cross_power_spectrum(fft1: &Array2<Complex<f32>>, fft2: &Array2<Complex<f32>>) -> Array2<Complex<f32>> {
    let (height, width) = fft1.dim();
    let mut result = Array2::zeros((height, width));
    
    for y in 0..height {
        for x in 0..width {
            let f1 = fft1[[y, x]];
            let f2_conj = fft2[[y, x]].conj();
            let product = f1 * f2_conj;
            let magnitude = product.norm();
            
            if magnitude > 1e-10 {
                result[[y, x]] = product / magnitude;
            } else {
                result[[y, x]] = Complex::new(0.0, 0.0);
            }
        }
    }
    
    result
}

fn find_correlation_peak(correlation: &Array2<Complex<f32>>) -> (usize, usize) {
    let (height, width) = correlation.dim();
    let mut max_val = f32::NEG_INFINITY;
    let mut peak_x = 0;
    let mut peak_y = 0;
    
    for y in 0..height {
        for x in 0..width {
            let magnitude = correlation[[y, x]].norm();
            if magnitude > max_val {
                max_val = magnitude;
                peak_x = x;
                peak_y = y;
            }
        }
    }
    
    (peak_x, peak_y)
}