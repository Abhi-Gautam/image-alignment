use crate::config::AugmentationConfig;
use opencv::core::*;
use rand::Rng;
use std::f32::consts::PI;

pub struct ImageManipulator {
    config: AugmentationConfig,
}

impl ImageManipulator {
    pub fn new(config: AugmentationConfig) -> Self {
        Self { config }
    }

    pub fn apply_noise(&self, image: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        if self.config.noise.enable_gaussian {
            self.apply_gaussian_noise(image, &mut rng)?;
        }

        if self.config.noise.enable_salt_pepper {
            self.apply_salt_pepper_noise(image, &mut rng)?;
        }

        if self.config.noise.enable_uniform {
            self.apply_uniform_noise(image, &mut rng)?;
        }

        Ok(())
    }

    fn apply_gaussian_noise(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let std_dev = rng.gen_range(self.config.noise.gaussian_std_range.0..=self.config.noise.gaussian_std_range.1);
        
        let mut noise = Mat::default();
        let size = image.size()?;
        let _noise_mat = Mat::zeros(size.height, size.width, CV_32F)?;
        
        opencv::core::randn(&mut noise, &Scalar::all(0.0), &Scalar::all(std_dev as f64))?;
        
        let mut image_f32 = Mat::default();
        image.convert_to(&mut image_f32, CV_32F, 1.0, 0.0)?;
        
        let mut result = Mat::default();
        opencv::core::add(&image_f32, &noise, &mut result, &Mat::default(), -1)?;
        image_f32 = result;
        
        image_f32.convert_to(image, image.typ(), 1.0, 0.0)?;
        
        Ok(())
    }

    fn apply_salt_pepper_noise(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let size = image.size()?;
        let total_pixels = (size.height * size.width) as f32;
        let num_noisy_pixels = (total_pixels * self.config.noise.salt_pepper_ratio) as i32;

        for _ in 0..num_noisy_pixels {
            let x = rng.gen_range(0..size.width);
            let y = rng.gen_range(0..size.height);
            
            let value = if rng.gen_bool(0.5) { 255u8 } else { 0u8 };
            
            unsafe {
                let ptr = image.ptr_2d_mut(y, x)?;
                *ptr = value;
            }
        }

        Ok(())
    }

    fn apply_uniform_noise(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let size = image.size()?;
        
        for y in 0..size.height {
            for x in 0..size.width {
                let noise = rng.gen_range(self.config.noise.uniform_range.0..=self.config.noise.uniform_range.1);
                
                unsafe {
                    let ptr = image.ptr_2d_mut(y, x)?;
                    let new_value = (*ptr as f32 + noise).clamp(0.0, 255.0) as u8;
                    *ptr = new_value;
                }
            }
        }

        Ok(())
    }

    pub fn apply_blur(&self, image: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        if self.config.blur.enable_gaussian {
            self.apply_gaussian_blur(image, &mut rng)?;
        }

        if self.config.blur.enable_motion {
            self.apply_motion_blur(image, &mut rng)?;
        }

        Ok(())
    }

    fn apply_gaussian_blur(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let kernel_size = rng.gen_range(self.config.blur.gaussian_kernel_range.0..=self.config.blur.gaussian_kernel_range.1);
        let sigma = rng.gen_range(self.config.blur.gaussian_sigma_range.0..=self.config.blur.gaussian_sigma_range.1);
        
        let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };
        
        let mut blurred = Mat::default();
        opencv::imgproc::gaussian_blur(
            image,
            &mut blurred,
            Size::new(kernel_size, kernel_size),
            sigma as f64,
            sigma as f64,
            opencv::core::BORDER_DEFAULT,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        
        *image = blurred;
        Ok(())
    }

    fn apply_motion_blur(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let length = rng.gen_range(self.config.blur.motion_length_range.0..=self.config.blur.motion_length_range.1);
        let angle = rng.gen_range(self.config.blur.motion_angle_range.0..=self.config.blur.motion_angle_range.1);
        
        let kernel = self.create_motion_blur_kernel(length, angle)?;
        
        let mut blurred = Mat::default();
        opencv::imgproc::filter_2d(
            image,
            &mut blurred,
            -1,
            &kernel,
            Point::new(-1, -1),
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;
        
        *image = blurred;
        Ok(())
    }

    fn create_motion_blur_kernel(&self, length: i32, angle: f32) -> Result<Mat, Box<dyn std::error::Error>> {
        let size = length;
        let mut kernel = Mat::zeros(size, size, CV_32F)?.to_mat()?;
        
        let center = size as f32 / 2.0;
        let angle_rad = angle * PI / 180.0;
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();
        
        for i in 0..length {
            let t = i as f32 - center;
            let x = (center + t * cos_angle).round() as i32;
            let y = (center + t * sin_angle).round() as i32;
            
            if x >= 0 && x < size && y >= 0 && y < size {
                *kernel.at_2d_mut::<f32>(y, x)? = 1.0 / length as f32;
            }
        }
        
        Ok(kernel)
    }

    pub fn apply_lighting(&self, image: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        if self.config.lighting.enable_brightness {
            self.adjust_brightness(image, &mut rng)?;
        }

        if self.config.lighting.enable_contrast {
            self.adjust_contrast(image, &mut rng)?;
        }

        if self.config.lighting.enable_gamma {
            self.adjust_gamma(image, &mut rng)?;
        }

        if self.config.lighting.enable_exposure {
            self.adjust_exposure(image, &mut rng)?;
        }

        Ok(())
    }

    fn adjust_brightness(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let brightness = rng.gen_range(self.config.lighting.brightness_range.0..=self.config.lighting.brightness_range.1);
        
        let mut adjusted = Mat::default();
        image.convert_to(&mut adjusted, -1, 1.0, brightness as f64)?;
        *image = adjusted;
        
        Ok(())
    }

    fn adjust_contrast(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let contrast = rng.gen_range(self.config.lighting.contrast_range.0..=self.config.lighting.contrast_range.1);
        
        let mut adjusted = Mat::default();
        image.convert_to(&mut adjusted, -1, contrast as f64, 0.0)?;
        *image = adjusted;
        
        Ok(())
    }

    fn adjust_gamma(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let gamma = rng.gen_range(self.config.lighting.gamma_range.0..=self.config.lighting.gamma_range.1);
        
        let mut normalized = Mat::default();
        image.convert_to(&mut normalized, CV_32F, 1.0 / 255.0, 0.0)?;
        
        let mut result = Mat::default();
        opencv::core::pow(&normalized, gamma as f64, &mut result)?;
        normalized = result;
        
        normalized.convert_to(image, image.typ(), 255.0, 0.0)?;
        
        Ok(())
    }

    fn adjust_exposure(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let exposure = rng.gen_range(self.config.lighting.exposure_range.0..=self.config.lighting.exposure_range.1);
        let scale = 2.0_f32.powf(exposure);
        
        let mut adjusted = Mat::default();
        image.convert_to(&mut adjusted, -1, scale as f64, 0.0)?;
        *image = adjusted;
        
        Ok(())
    }

    pub fn apply_geometric_transform(&self, image: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();
        let size = image.size()?;
        let center = Point2f::new(size.width as f32 / 2.0, size.height as f32 / 2.0);

        let mut transform_matrix = Mat::eye(2, 3, CV_32F)?.to_mat()?;

        if self.config.geometric.enable_rotation {
            let angle = rng.gen_range(self.config.geometric.rotation_range.0..=self.config.geometric.rotation_range.1);
            let rotation_matrix = opencv::imgproc::get_rotation_matrix_2d(center, angle as f64, 1.0)?;
            let mut temp_result = Mat::default();
            opencv::core::multiply(&transform_matrix, &rotation_matrix, &mut temp_result, 1.0, -1)?;
            transform_matrix = temp_result;
        }

        if self.config.geometric.enable_scale {
            let scale = rng.gen_range(self.config.geometric.scale_range.0..=self.config.geometric.scale_range.1);
            let scale_matrix = opencv::imgproc::get_rotation_matrix_2d(center, 0.0, scale as f64)?;
            let mut temp_result = Mat::default();
            opencv::core::multiply(&transform_matrix, &scale_matrix, &mut temp_result, 1.0, -1)?;
            transform_matrix = temp_result;
        }

        if self.config.geometric.enable_translation {
            let tx = rng.gen_range(self.config.geometric.translation_range.0..=self.config.geometric.translation_range.1);
            let ty = rng.gen_range(self.config.geometric.translation_range.0..=self.config.geometric.translation_range.1);
            
            *transform_matrix.at_2d_mut::<f32>(0, 2)? += tx;
            *transform_matrix.at_2d_mut::<f32>(1, 2)? += ty;
        }

        let mut transformed = Mat::default();
        opencv::imgproc::warp_affine(
            image,
            &mut transformed,
            &transform_matrix,
            size,
            opencv::imgproc::INTER_LINEAR,
            opencv::core::BORDER_REPLICATE,
            Scalar::default(),
        )?;

        *image = transformed;
        Ok(())
    }

    pub fn apply_distortion(&self, image: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        if self.config.distortion.enable_barrel {
            self.apply_barrel_distortion(image, &mut rng)?;
        }

        if self.config.distortion.enable_pincushion {
            self.apply_pincushion_distortion(image, &mut rng)?;
        }

        if self.config.distortion.enable_perspective {
            self.apply_perspective_distortion(image, &mut rng)?;
        }

        Ok(())
    }

    fn apply_barrel_distortion(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let k1 = rng.gen_range(self.config.distortion.barrel_distortion_range.0..=self.config.distortion.barrel_distortion_range.1);
        self.apply_radial_distortion(image, k1, 0.0)
    }

    fn apply_pincushion_distortion(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let k1 = rng.gen_range(self.config.distortion.pincushion_distortion_range.0..=self.config.distortion.pincushion_distortion_range.1);
        self.apply_radial_distortion(image, k1, 0.0)
    }

    fn apply_radial_distortion(&self, image: &mut Mat, k1: f32, k2: f32) -> Result<(), Box<dyn std::error::Error>> {
        let size = image.size()?;
        let center_x = size.width as f32 / 2.0;
        let center_y = size.height as f32 / 2.0;

        let mut distorted = Mat::default();
        image.copy_to(&mut distorted)?;

        for y in 0..size.height {
            for x in 0..size.width {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let r = (dx * dx + dy * dy).sqrt();
                let r_norm = r / (center_x.min(center_y));

                let distortion_factor = 1.0 + k1 * r_norm * r_norm + k2 * r_norm * r_norm * r_norm * r_norm;
                
                let src_x = center_x + dx / distortion_factor;
                let src_y = center_y + dy / distortion_factor;

                if src_x >= 0.0 && src_x < size.width as f32 && src_y >= 0.0 && src_y < size.height as f32 {
                    let src_x_int = src_x as i32;
                    let src_y_int = src_y as i32;
                    
                    if src_x_int >= 0 && src_x_int < size.width && src_y_int >= 0 && src_y_int < size.height {
                        let src_pixel = *image.at_2d::<u8>(src_y_int, src_x_int)?;
                        *distorted.at_2d_mut::<u8>(y, x)? = src_pixel;
                    }
                }
            }
        }

        *image = distorted;
        Ok(())
    }

    fn apply_perspective_distortion(&self, image: &mut Mat, rng: &mut impl Rng) -> Result<(), Box<dyn std::error::Error>> {
        let size = image.size()?;
        let strength = self.config.distortion.perspective_distortion_strength;
        
        let offset = (size.width as f32 * strength * rng.gen_range(-1.0..=1.0)) as i32;
        
        let src_points = vec![
            Point2f::new(0.0, 0.0),
            Point2f::new(size.width as f32, 0.0),
            Point2f::new(size.width as f32, size.height as f32),
            Point2f::new(0.0, size.height as f32),
        ];

        let dst_points = vec![
            Point2f::new(offset as f32, 0.0),
            Point2f::new(size.width as f32 - offset as f32, 0.0),
            Point2f::new(size.width as f32, size.height as f32),
            Point2f::new(0.0, size.height as f32),
        ];

        let perspective_matrix = opencv::imgproc::get_perspective_transform(
            &opencv::core::Vector::<opencv::core::Point2f>::from_iter(src_points),
            &opencv::core::Vector::<opencv::core::Point2f>::from_iter(dst_points),
            opencv::core::DECOMP_LU,
        )?;

        let mut transformed = Mat::default();
        opencv::imgproc::warp_perspective(
            image,
            &mut transformed,
            &perspective_matrix,
            size,
            opencv::imgproc::INTER_LINEAR,
            opencv::core::BORDER_REPLICATE,
            Scalar::default(),
        )?;

        *image = transformed;
        Ok(())
    }

    pub fn apply_all_augmentations(&self, image: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        self.apply_noise(image)?;
        self.apply_blur(image)?;
        self.apply_lighting(image)?;
        self.apply_geometric_transform(image)?;
        self.apply_distortion(image)?;
        Ok(())
    }

    pub fn apply_selective_augmentations(&self, image: &mut Mat, augmentations: &[AugmentationType]) -> Result<(), Box<dyn std::error::Error>> {
        for aug_type in augmentations {
            match aug_type {
                AugmentationType::Noise => self.apply_noise(image)?,
                AugmentationType::Blur => self.apply_blur(image)?,
                AugmentationType::Lighting => self.apply_lighting(image)?,
                AugmentationType::Geometric => self.apply_geometric_transform(image)?,
                AugmentationType::Distortion => self.apply_distortion(image)?,
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AugmentationType {
    Noise,
    Blur,
    Lighting,
    Geometric,
    Distortion,
}

impl std::str::FromStr for AugmentationType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "noise" => Ok(AugmentationType::Noise),
            "blur" => Ok(AugmentationType::Blur),
            "lighting" => Ok(AugmentationType::Lighting),
            "geometric" => Ok(AugmentationType::Geometric),
            "distortion" => Ok(AugmentationType::Distortion),
            _ => Err(format!("Unknown augmentation type: {}", s)),
        }
    }
}