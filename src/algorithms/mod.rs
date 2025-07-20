// Legacy custom implementations (will be removed)
pub mod orb_matching;
pub mod fast_orb;
pub mod orb_improved;
pub mod phase_correlation;
pub mod template_matching;

// New OpenCV-based implementations
pub mod opencv_template;
pub mod opencv_orb;

pub use orb_matching::*;
pub use fast_orb::*;
pub use orb_improved::*;
pub use phase_correlation::*;
pub use template_matching::*;
pub use opencv_template::*;
// Qualified import to avoid Keypoint name conflict
pub use opencv_orb::{OpenCVORB, DMatch};

use crate::AlignmentResult;
use image::GrayImage;

pub trait AlignmentAlgorithm {
    fn align(&self, template: &GrayImage, target: &GrayImage) -> crate::Result<AlignmentResult>;
    fn name(&self) -> &'static str;
}