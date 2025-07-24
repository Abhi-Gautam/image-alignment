// OpenCV-based implementations only
pub mod opencv_akaze;
pub mod opencv_ecc;
pub mod opencv_orb;
pub mod opencv_sift;
pub mod opencv_template;

pub use opencv_akaze::*;
pub use opencv_ecc::*;
pub use opencv_orb::*;
pub use opencv_sift::*;
pub use opencv_template::*;
