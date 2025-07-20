use image::{GrayImage, ImageBuffer};
use std::path::{Path, PathBuf};

pub struct PatchExtractor;

impl PatchExtractor {
    /// Extract a patch from an image at the specified location
    pub fn extract_patch(
        source_image: &GrayImage,
        center_x: u32,
        center_y: u32,
        patch_size: u32,
    ) -> crate::Result<GrayImage> {
        let half_size = patch_size / 2;
        
        // Check bounds
        if center_x < half_size || center_y < half_size {
            return Err(anyhow::anyhow!("Patch center too close to image edge"));
        }
        
        if center_x + half_size >= source_image.width() || center_y + half_size >= source_image.height() {
            return Err(anyhow::anyhow!("Patch extends beyond image bounds"));
        }
        
        let start_x = center_x - half_size;
        let start_y = center_y - half_size;
        
        let mut patch = ImageBuffer::new(patch_size, patch_size);
        
        for y in 0..patch_size {
            for x in 0..patch_size {
                let src_x = start_x + x;
                let src_y = start_y + y;
                let pixel = source_image.get_pixel(src_x, src_y);
                patch.put_pixel(x, y, *pixel);
            }
        }
        
        Ok(patch)
    }
    
    /// Extract multiple patches from an image with good feature content
    pub fn extract_good_patches(
        source_image: &GrayImage,
        patch_size: u32,
        count: u32,
    ) -> crate::Result<Vec<(GrayImage, u32, u32)>> {
        let mut patches = Vec::new();
        let half_size = patch_size / 2;
        let min_variance = 100.0; // Minimum variance to ensure patch has features
        
        let mut attempts = 0;
        let max_attempts = count * 10; // Try up to 10x the requested count
        
        while patches.len() < count as usize && attempts < max_attempts {
            // Random position (avoiding edges)
            let center_x = half_size + (rand::random::<u32>() % (source_image.width() - patch_size));
            let center_y = half_size + (rand::random::<u32>() % (source_image.height() - patch_size));
            
            if let Ok(patch) = Self::extract_patch(source_image, center_x, center_y, patch_size) {
                // Check if patch has sufficient contrast/features
                let variance = Self::calculate_variance(&patch);
                if variance > min_variance {
                    patches.push((patch, center_x, center_y));
                }
            }
            
            attempts += 1;
        }
        
        if patches.is_empty() {
            return Err(anyhow::anyhow!("Could not find any good patches in the image"));
        }
        
        Ok(patches)
    }
    
    /// Calculate the variance of pixel intensities (measure of contrast)
    fn calculate_variance(image: &GrayImage) -> f32 {
        let pixels: Vec<f32> = image.pixels().map(|p| p[0] as f32).collect();
        let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
        let variance = pixels.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f32>() / pixels.len() as f32;
        variance
    }
    
    /// Save patches to files with descriptive names
    pub fn save_patches(
        patches: &[(GrayImage, u32, u32)],
        base_name: &str,
        output_dir: &Path,
    ) -> crate::Result<Vec<PathBuf>> {
        std::fs::create_dir_all(output_dir)?;
        let mut saved_paths = Vec::new();
        
        for (i, (patch, center_x, center_y)) in patches.iter().enumerate() {
            let filename = format!("{}_patch_{:02}_at_{}x{}.png", base_name, i, center_x, center_y);
            let filepath = output_dir.join(filename);
            patch.save(&filepath)?;
            saved_paths.push(filepath);
        }
        
        Ok(saved_paths)
    }
}