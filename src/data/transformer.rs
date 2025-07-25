use image::{GrayImage, ImageBuffer, Luma};
use std::f32::consts::PI;

pub struct ImageTransformer;

impl ImageTransformer {
    /// Rotate an image by the specified angle (in degrees)
    pub fn rotate(image: &GrayImage, angle_degrees: f32) -> crate::Result<GrayImage> {
        let angle_rad = angle_degrees * PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let width = image.width();
        let height = image.height();
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        let mut rotated = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                // Translate to center
                let x_centered = x as f32 - center_x;
                let y_centered = y as f32 - center_y;

                // Apply rotation (inverse transform)
                let x_rot = x_centered * cos_a + y_centered * sin_a + center_x;
                let y_rot = -x_centered * sin_a + y_centered * cos_a + center_y;

                // Bilinear interpolation
                let pixel_value = Self::bilinear_interpolate(image, x_rot, y_rot);
                rotated.put_pixel(x, y, Luma([pixel_value]));
            }
        }

        Ok(rotated)
    }

    /// Translate an image by dx, dy pixels
    pub fn translate(image: &GrayImage, dx: i32, dy: i32) -> crate::Result<GrayImage> {
        let width = image.width();
        let height = image.height();
        let mut translated = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let src_x = x as i32 - dx;
                let src_y = y as i32 - dy;

                let pixel_value =
                    if src_x >= 0 && src_x < width as i32 && src_y >= 0 && src_y < height as i32 {
                        image.get_pixel(src_x as u32, src_y as u32)[0]
                    } else {
                        0 // Black for out-of-bounds
                    };

                translated.put_pixel(x, y, Luma([pixel_value]));
            }
        }

        Ok(translated)
    }

    /// Scale an image by the specified factor
    pub fn scale(image: &GrayImage, scale_factor: f32) -> crate::Result<GrayImage> {
        let width = image.width();
        let height = image.height();
        let mut scaled = ImageBuffer::new(width, height);

        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        for y in 0..height {
            for x in 0..width {
                // Translate to center, scale, then translate back
                let x_centered = (x as f32 - center_x) / scale_factor + center_x;
                let y_centered = (y as f32 - center_y) / scale_factor + center_y;

                let pixel_value = Self::bilinear_interpolate(image, x_centered, y_centered);
                scaled.put_pixel(x, y, Luma([pixel_value]));
            }
        }

        Ok(scaled)
    }

    /// Apply rotation and translation together
    pub fn rotate_and_translate(
        image: &GrayImage,
        angle_degrees: f32,
        dx: i32,
        dy: i32,
    ) -> crate::Result<GrayImage> {
        let rotated = Self::rotate(image, angle_degrees)?;
        Self::translate(&rotated, dx, dy)
    }

    /// Bilinear interpolation for smooth transformations
    fn bilinear_interpolate(image: &GrayImage, x: f32, y: f32) -> u8 {
        let width = image.width();
        let height = image.height();

        if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
            return 0; // Out of bounds
        }

        let x1 = x.floor() as u32;
        let y1 = y.floor() as u32;
        let x2 = (x1 + 1).min(width - 1);
        let y2 = (y1 + 1).min(height - 1);

        let fx = x - x1 as f32;
        let fy = y - y1 as f32;

        let p11 = image.get_pixel(x1, y1)[0] as f32;
        let p12 = image.get_pixel(x1, y2)[0] as f32;
        let p21 = image.get_pixel(x2, y1)[0] as f32;
        let p22 = image.get_pixel(x2, y2)[0] as f32;

        let interpolated = p11 * (1.0 - fx) * (1.0 - fy)
            + p21 * fx * (1.0 - fy)
            + p12 * (1.0 - fx) * fy
            + p22 * fx * fy;

        interpolated.round() as u8
    }
}

/// Ground truth transformation for validation
#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub rotation_degrees: f32,
    pub translation_x: i32,
    pub translation_y: i32,
    pub scale_factor: f32,
}

impl Default for GroundTruth {
    fn default() -> Self {
        Self::new()
    }
}

impl GroundTruth {
    pub fn new() -> Self {
        Self {
            rotation_degrees: 0.0,
            translation_x: 0,
            translation_y: 0,
            scale_factor: 1.0,
        }
    }

    pub fn rotation(angle: f32) -> Self {
        Self {
            rotation_degrees: angle,
            translation_x: 0,
            translation_y: 0,
            scale_factor: 1.0,
        }
    }

    pub fn translation(dx: i32, dy: i32) -> Self {
        Self {
            rotation_degrees: 0.0,
            translation_x: dx,
            translation_y: dy,
            scale_factor: 1.0,
        }
    }
}
