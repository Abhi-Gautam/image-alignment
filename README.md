# Image Alignment System

High-performance image alignment system for semiconductor wafer inspection using OpenCV-based algorithms and real-time processing capabilities.

## Overview

A production-ready Rust-based image alignment system designed for semiconductor wafer inspection. The system provides sub-pixel accuracy alignment of template images to larger SEM images with comprehensive performance analysis and logging capabilities.

### Key Features

- **Multiple OpenCV Algorithms**: SIFT, ORB, AKAZE, and Template Matching algorithms
- **Real-time Performance**: Sub-50ms processing time for production workflows
- **Sub-pixel Accuracy**: Precise alignment critical for semiconductor inspection
- **Interactive Dashboard**: Web-based visualization and analysis interface
- **Comprehensive Logging**: Per-test algorithm execution logs with ground truth analysis
- **Configuration-Driven**: Fully configurable algorithm parameters via TOML
- **Batch Processing**: Automated testing with multiple patch sizes and scenarios

## Quick Start

### Prerequisites

- Rust 1.70 or later
- OpenCV 4.x installed on your system
- pkg-config (for OpenCV linking)

### Installation

```bash
# Clone repository
git clone https://github.com/Abhi-Gautam/image-alignment
cd image-alignment

# Set up OpenCV environment
./setup_opencv_env.sh

# Build in release mode
cargo build --release

# Run tests to verify installation
cargo test --release
```

### Basic Usage

```bash
# Run comprehensive visual tests
cargo run --release -- visual-test \
  --sem-image datasets/wafer_image.png \
  --patch-sizes 64,128,192 \
  --scenarios clean,rotation_10deg,gaussian_noise \
  --output results/

# Launch interactive dashboard
cargo run --release -- dashboard --port 8080

# Run with custom configuration
cargo run --release -- --config config.toml visual-test \
  --sem-image image.png \
  --output results/
```

## Architecture

### Supported Algorithms

| Algorithm | Method | Speed | Best Use Case | Key Features |
|-----------|--------|-------|---------------|--------------|
| **OpenCV-SIFT** | Feature-based | ~40ms | High accuracy needed | Scale/rotation invariant, patented algorithm |
| **OpenCV-ORB** | Feature-based | ~25ms | Real-time applications | Free alternative to SIFT, good performance |
| **OpenCV-AKAZE** | Feature-based | ~30ms | Balanced performance | Non-linear scale space, robust features |
| **OpenCV-NCC** | Template matching | ~15ms | Clean images | Normalized cross-correlation |
| **OpenCV-SSD** | Template matching | ~10ms | Speed critical | Sum of squared differences |

### System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   CLI/Config    │────▶│   Pipeline   │────▶│   Algorithms    │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Dashboard     │     │   Logging    │     │  Visualization  │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Configuration

The system uses a TOML configuration file for all algorithm parameters. See `config.toml` for the default configuration.

```toml
# Example configuration snippet
[algorithms.sift]
n_features = 1000
contrast_threshold = 0.04
edge_threshold = 10.0
sigma = 1.6

[algorithms.orb]
n_features = 1000
scale_factor = 1.2
n_levels = 8
edge_threshold = 31
```

## Dashboard Interface

The interactive dashboard provides:

- **Visual Results**: Side-by-side comparison of alignment results
- **Performance Metrics**: Algorithm execution times and accuracy measurements
- **Multi-Select Filters**: Filter results by algorithm, patch size, and test scenario
- **Algorithm Logs**: Detailed per-test execution logs with ground truth analysis
- **Session Management**: Track and compare multiple test sessions

### Accessing the Dashboard

```bash
# Start the dashboard server
cargo run --release -- dashboard --port 8080

# Open in browser
http://localhost:8080
```

## Test Scenarios

The system supports various test scenarios to evaluate algorithm robustness:

| Scenario | Description | Parameters |
|----------|-------------|------------|
| `clean` | No transformation | Original patch |
| `translation_5px` | Small translation | ±5 pixels |
| `translation_10px` | Medium translation | ±10 pixels |
| `rotation_10deg` | Small rotation | 10° rotation |
| `rotation_30deg` | Large rotation | 30° rotation |
| `gaussian_noise` | Additive noise | σ=5.0 |
| `salt_pepper` | Impulse noise | density=0.005 |
| `gaussian_blur` | Motion blur | σ=1.5 |
| `brightness_change` | Illumination | ±20 intensity |
| `scale_120` | Scale change | 1.2x scale |

## Algorithm Execution Logs

Each test generates detailed algorithm execution logs containing:

```
=== Ground Truth Information ===
Original Patch Location: (256, 128)
Original Patch Size: 64x64
Patch Center: (288, 160)

=== Test Scenario Details ===
Noise Type: None
Rotation: 10°
Translation: (0, 0)
Expected Location After Transform: (256, 128)

=== Algorithm Results ===
Detected Location: (258, 127)
Confidence: 0.922
Execution Time: 42.4ms

=== Error Analysis ===
Detection Error X: 2 pixels
Detection Error Y: -1 pixels
Total Translation Error: 2.24 pixels
Success Threshold: <= 10 pixels
Test Result: PASSED
```

## System Requirements

### Hardware Requirements
- **CPU**: Modern x86_64 or ARM64 processor
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 500MB for dependencies and build artifacts
- **GPU**: Optional, not currently utilized

### Software Requirements
- **Rust**: 1.70 or later
- **OpenCV**: 4.x with development headers
- **pkg-config**: For library linking
- **OS**: Linux, macOS, Windows (with OpenCV properly configured)

### Key Dependencies
- `opencv` - Computer vision algorithms
- `image` - Image I/O and basic operations
- `axum` - Web framework for dashboard
- `tracing` - Structured logging framework
- `serde` - Serialization for configuration and data
- `clap` - Command-line interface

## Production Deployment

### Performance Optimization

```bash
# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run with optimized thread pool
RAYON_NUM_THREADS=8 cargo run --release -- visual-test ...
```

### Monitoring and Logging

The system provides comprehensive logging with different levels:

```bash
# Set logging level
RUST_LOG=info cargo run --release -- dashboard

# Enable debug logging for specific modules
RUST_LOG=image_alignment::algorithms=debug cargo run --release ...
```

### Integration

The system can be integrated into existing workflows:

1. **CLI Integration**: Use JSON output for scripting
2. **REST API**: Dashboard server provides API endpoints
3. **Library Usage**: Import as Rust crate for custom applications

## API Endpoints

The dashboard server provides REST API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data` | GET | Get filtered test results |
| `/api/sessions` | GET | List all test sessions |
| `/api/sessions/:id` | GET | Get specific session |
| `/api/algorithms/summary` | GET | Algorithm performance summary |
| `/api/run-visual-test` | POST | Execute visual test |
| `/api/test-logs/:session/:test` | GET | Get algorithm execution log |
| `/api/image/*path` | GET | Serve result images |

## Troubleshooting

### OpenCV Installation Issues

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev pkg-config

# macOS
brew install opencv pkg-config

# Verify installation
pkg-config --modversion opencv4
```

### Build Issues

```bash
# Clear build cache
cargo clean

# Rebuild with verbose output
cargo build --release --verbose

# Check OpenCV linking
pkg-config --libs opencv4
```

### Performance Issues

- Ensure release build is used
- Check image sizes (larger images = slower processing)
- Monitor CPU usage during execution
- Consider reducing patch sizes or test scenarios

## Project Structure

```
image-alignment/
├── src/
│   ├── algorithms/      # OpenCV algorithm implementations
│   ├── config/          # Configuration management
│   ├── dashboard/       # Web dashboard and API
│   ├── logging/         # Structured logging system
│   ├── pipeline/        # Processing pipeline framework
│   ├── visualization/   # Visual testing and reporting
│   └── main.rs          # CLI entry point
├── config.toml          # Default configuration
├── datasets/            # Test images and data
└── results/             # Output directory
```

## Development

### Running Tests

```bash
# Run all tests
cargo test --release

# Run specific test module
cargo test --release algorithms

# Run with output
cargo test --release -- --nocapture
```

### Contributing

Contributions are welcome. Please ensure:

1. Code follows Rust idioms and conventions
2. All tests pass before submitting PR
3. New features include appropriate tests
4. Documentation is updated as needed

### Building Documentation

```bash
# Generate and open documentation
cargo doc --open

# Include private items
cargo doc --document-private-items
```

## License

MIT License - see LICENSE file for details.

## Contact

- Repository: https://github.com/Abhi-Gautam/image-alignment
- Issues: https://github.com/Abhi-Gautam/image-alignment/issues