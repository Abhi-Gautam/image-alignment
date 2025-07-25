use crate::AlignmentResult;

pub fn calculate_translation_error(result: &AlignmentResult, ground_truth: (f32, f32)) -> f32 {
    let (dx, dy) = (
        result.translation.0 - ground_truth.0,
        result.translation.1 - ground_truth.1,
    );
    (dx * dx + dy * dy).sqrt()
}

pub fn calculate_rotation_error(result: &AlignmentResult, ground_truth: f32) -> f32 {
    (result.rotation - ground_truth).abs()
}

pub fn calculate_scale_error(result: &AlignmentResult, ground_truth: f32) -> f32 {
    (result.scale - ground_truth).abs()
}
