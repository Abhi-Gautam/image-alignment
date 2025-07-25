use crate::pipeline::{AlignmentAlgorithm, AlignmentResult, ComplexityClass};
use crate::Result;
use opencv::core::{Mat, Rect};
use opencv::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Coarse-to-fine alignment strategy
pub struct CoarseToFineAlgorithm {
    coarse: Box<dyn AlignmentAlgorithm>,
    fine: Box<dyn AlignmentAlgorithm>,
    search_radius: i32,
}

impl CoarseToFineAlgorithm {
    pub fn new(coarse: Box<dyn AlignmentAlgorithm>, fine: Box<dyn AlignmentAlgorithm>) -> Self {
        Self {
            coarse,
            fine,
            search_radius: 20, // Default search radius for fine alignment
        }
    }

    pub fn with_search_radius(mut self, radius: i32) -> Self {
        self.search_radius = radius;
        self
    }
}

impl AlignmentAlgorithm for CoarseToFineAlgorithm {
    fn name(&self) -> &str {
        "CoarseToFine"
    }

    fn align(&self, search_image: &Mat, patch: &Mat) -> Result<AlignmentResult> {
        let start = Instant::now();

        // Step 1: Coarse alignment
        let coarse_result = self.coarse.align(search_image, patch)?;

        // Step 2: Extract region around coarse result for fine alignment
        let coarse_rect: Rect = coarse_result.location.clone().into();
        let roi_x = (coarse_rect.x - self.search_radius).max(0);
        let roi_y = (coarse_rect.y - self.search_radius).max(0);
        let roi_width =
            (coarse_rect.width + 2 * self.search_radius).min(search_image.cols() - roi_x);
        let roi_height =
            (coarse_rect.height + 2 * self.search_radius).min(search_image.rows() - roi_y);

        let roi_rect = Rect::new(roi_x, roi_y, roi_width, roi_height);
        let roi = search_image.roi(roi_rect)?;

        // Step 3: Fine alignment within ROI
        let roi_mat = roi.clone_pointee();
        let mut fine_result = self.fine.align(&roi_mat, patch)?;

        // Step 4: Adjust fine result coordinates to global space
        let fine_rect: Rect = fine_result.location.clone().into();
        fine_result.location = Rect::new(
            fine_rect.x + roi_x,
            fine_rect.y + roi_y,
            fine_rect.width,
            fine_rect.height,
        )
        .into();

        // Update metadata
        fine_result.algorithm_name = format!(
            "CoarseToFine({} -> {})",
            self.coarse.name(),
            self.fine.name()
        );
        fine_result.execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        fine_result.metadata.insert(
            "coarse_score".to_string(),
            serde_json::Value::from(coarse_result.score),
        );
        fine_result.metadata.insert(
            "coarse_location".to_string(),
            serde_json::to_value(&coarse_result.location)?,
        );

        Ok(fine_result)
    }

    fn estimated_complexity(&self) -> ComplexityClass {
        // Return the higher complexity of the two
        match (
            self.coarse.estimated_complexity(),
            self.fine.estimated_complexity(),
        ) {
            (ComplexityClass::High, _) | (_, ComplexityClass::High) => ComplexityClass::High,
            (ComplexityClass::Medium, _) | (_, ComplexityClass::Medium) => ComplexityClass::Medium,
            _ => ComplexityClass::Low,
        }
    }
}

/// Voting strategies for ensemble algorithms
#[derive(Debug, Clone)]
pub enum VotingStrategy {
    /// Simple majority voting
    Majority,
    /// Weighted voting based on confidence scores
    WeightedConfidence,
    /// Take the result with highest confidence
    MaxConfidence,
    /// Average the positions (for continuous outputs)
    Average,
}

/// Ensemble algorithm that combines multiple algorithms
pub struct EnsembleAlgorithm {
    algorithms: Vec<Box<dyn AlignmentAlgorithm>>,
    voting_strategy: VotingStrategy,
}

impl EnsembleAlgorithm {
    pub fn new(algorithms: Vec<Box<dyn AlignmentAlgorithm>>) -> Self {
        Self {
            algorithms,
            voting_strategy: VotingStrategy::WeightedConfidence,
        }
    }

    pub fn with_voting_strategy(mut self, strategy: VotingStrategy) -> Self {
        self.voting_strategy = strategy;
        self
    }
}

impl AlignmentAlgorithm for EnsembleAlgorithm {
    fn name(&self) -> &str {
        "Ensemble"
    }

    fn align(&self, search_image: &Mat, patch: &Mat) -> Result<AlignmentResult> {
        let start = Instant::now();

        // Run all algorithms
        let mut results = Vec::new();
        for algo in &self.algorithms {
            match algo.align(search_image, patch) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Log error but continue with other algorithms
                    eprintln!("Algorithm {} failed: {}", algo.name(), e);
                }
            }
        }

        if results.is_empty() {
            return Err(anyhow::anyhow!("All algorithms in ensemble failed"));
        }

        // Apply voting strategy
        let final_result = match self.voting_strategy {
            VotingStrategy::Majority => self.majority_vote(&results),
            VotingStrategy::WeightedConfidence => self.weighted_confidence_vote(&results),
            VotingStrategy::MaxConfidence => self.max_confidence_vote(&results),
            VotingStrategy::Average => self.average_vote(&results),
        }?;

        let mut result = final_result;
        result.algorithm_name = format!(
            "Ensemble({} algorithms, {:?} voting)",
            self.algorithms.len(),
            self.voting_strategy
        );
        result.execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }
}

impl EnsembleAlgorithm {
    fn majority_vote(&self, results: &[AlignmentResult]) -> Result<AlignmentResult> {
        // Group results by location (with some tolerance)
        let tolerance = 5; // pixels
        let mut location_votes: HashMap<(i32, i32), Vec<&AlignmentResult>> = HashMap::new();

        for result in results {
            let key = (
                result.location.x / tolerance * tolerance,
                result.location.y / tolerance * tolerance,
            );
            location_votes.entry(key).or_default().push(result);
        }

        // Find location with most votes
        let (_, best_group) = location_votes
            .iter()
            .max_by_key(|(_, votes)| votes.len())
            .ok_or_else(|| anyhow::anyhow!("No consensus in majority voting"))?;

        // Return result with highest confidence from best group
        best_group
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .map(|r| (*r).clone())
            .ok_or_else(|| anyhow::anyhow!("No valid result in majority group"))
    }

    fn weighted_confidence_vote(&self, results: &[AlignmentResult]) -> Result<AlignmentResult> {
        let total_confidence: f64 = results.iter().map(|r| r.confidence).sum();

        if total_confidence == 0.0 {
            return self.max_confidence_vote(results);
        }

        // Weighted average of positions
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut weighted_score = 0.0;

        for result in results {
            let weight = result.confidence / total_confidence;
            weighted_x += result.location.x as f64 * weight;
            weighted_y += result.location.y as f64 * weight;
            weighted_score += result.score * weight;
        }

        // Find the result closest to weighted average
        let target_x = weighted_x as i32;
        let target_y = weighted_y as i32;

        results
            .iter()
            .min_by_key(|r| {
                let dx = r.location.x - target_x;
                let dy = r.location.y - target_y;
                dx * dx + dy * dy
            })
            .map(|r| {
                let mut result = r.clone();
                result.score = weighted_score;
                result
            })
            .ok_or_else(|| anyhow::anyhow!("No valid result for weighted voting"))
    }

    fn max_confidence_vote(&self, results: &[AlignmentResult]) -> Result<AlignmentResult> {
        results
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No results to select from"))
    }

    fn average_vote(&self, results: &[AlignmentResult]) -> Result<AlignmentResult> {
        let n = results.len() as f64;

        let avg_x = results.iter().map(|r| r.location.x as f64).sum::<f64>() / n;
        let avg_y = results.iter().map(|r| r.location.y as f64).sum::<f64>() / n;
        let avg_score = results.iter().map(|r| r.score).sum::<f64>() / n;
        let avg_confidence = results.iter().map(|r| r.confidence).sum::<f64>() / n;

        // Use the first result as template and update with averages
        let mut result = results[0].clone();
        result.location.x = avg_x as i32;
        result.location.y = avg_y as i32;
        result.score = avg_score;
        result.confidence = avg_confidence;

        Ok(result)
    }
}
