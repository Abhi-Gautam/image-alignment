use crate::pipeline::AlignmentAlgorithm;
use instant::Instant;
use opencv::core::Mat;

pub struct BenchmarkRunner {
    pub algorithms: Vec<Box<dyn AlignmentAlgorithm>>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self {
            algorithms: Vec::new(),
        }
    }

    pub fn add_algorithm(&mut self, algorithm: Box<dyn AlignmentAlgorithm>) {
        self.algorithms.push(algorithm);
    }

    pub fn run_benchmark(
        &self,
        template: &Mat,
        target: &Mat,
    ) -> Vec<crate::pipeline::AlignmentResult> {
        let mut results = Vec::new();

        for algorithm in &self.algorithms {
            let start = Instant::now();
            match algorithm.align(template, target) {
                Ok(mut result) => {
                    result.execution_time_ms = start.elapsed().as_millis() as f64;
                    results.push(result);
                }
                Err(e) => {
                    log::error!("Algorithm {} failed: {}", algorithm.name(), e);
                }
            }
        }

        results
    }
}
