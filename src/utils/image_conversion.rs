use crate::Result;
use image::GrayImage;
use opencv::core::Mat;
use opencv::prelude::*;

/// Convert a GrayImage to an OpenCV Mat
pub fn grayimage_to_mat(image: &GrayImage) -> Result<Mat> {
    let (width, height) = image.dimensions();
    let data = image.as_raw();

    // Create Mat from raw data
    let mut mat = Mat::zeros(height as i32, width as i32, opencv::core::CV_8UC1)?.to_mat()?;

    // Copy data row by row
    for y in 0..height {
        for x in 0..width {
            let pixel = data[(y * width + x) as usize];
            *mat.at_2d_mut::<u8>(y as i32, x as i32)? = pixel;
        }
    }

    Ok(mat)
}

/// Convert an OpenCV Mat to a GrayImage
pub fn image_to_grayimage(mat: &Mat) -> Result<GrayImage> {
    let rows = mat.rows();
    let cols = mat.cols();

    let mut data = Vec::with_capacity((rows * cols) as usize);

    for y in 0..rows {
        for x in 0..cols {
            let pixel = mat.at_2d::<u8>(y, x)?;
            data.push(*pixel);
        }
    }

    let gray_image = GrayImage::from_raw(cols as u32, rows as u32, data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create GrayImage from Mat"))?;

    Ok(gray_image)
}

/// Load and validate image from path
pub fn load_image(path: &std::path::Path) -> Result<GrayImage> {
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "Image file does not exist: {}",
            path.display()
        ));
    }

    let img = image::open(path)?;
    let gray_img = img.to_luma8();

    validate_image_size(&gray_img)?;
    Ok(gray_img)
}

/// Validate that image has reasonable dimensions
pub fn validate_image_size(image: &GrayImage) -> Result<()> {
    validate_image_size_with_limits(image, 10, 10000)
}

/// Validate image size with custom limits
pub fn validate_image_size_with_limits(
    image: &GrayImage,
    min_size: u32,
    max_size: u32,
) -> Result<()> {
    let (width, height) = image.dimensions();

    if width < min_size || height < min_size {
        return Err(anyhow::anyhow!(
            "Image too small: {}x{}, minimum: {}x{}",
            width,
            height,
            min_size,
            min_size
        ));
    }

    if width > max_size || height > max_size {
        return Err(anyhow::anyhow!(
            "Image too large: {}x{}, maximum: {}x{}",
            width,
            height,
            max_size,
            max_size
        ));
    }

    Ok(())
}

