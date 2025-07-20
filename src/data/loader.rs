use image::{GrayImage, open};
use std::path::Path;

pub fn load_image<P: AsRef<Path>>(path: P) -> crate::Result<GrayImage> {
    let img = open(path)?;
    Ok(img.to_luma8())
}

pub fn validate_image_size(img: &GrayImage, min_size: u32) -> crate::Result<()> {
    if img.width() < min_size || img.height() < min_size {
        return Err(anyhow::anyhow!("Image too small: {}x{}, minimum: {}x{}", 
                          img.width(), img.height(), min_size, min_size));
    }
    Ok(())
}