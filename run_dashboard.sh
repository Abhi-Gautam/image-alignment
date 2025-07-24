#!/bin/bash
set -e

echo "🧪 Running SEM Image Alignment Tests and Dashboard"

# Clean up previous results
echo "🧹 Cleaning up previous results..."
rm -rf results/dashboard_tests

# Run visual tests
echo "📊 Running visual tests..."
cargo run --bin align visual-test \
    --sem-image "./datasets/images/valid/19140_image1_jpg.rf.9cfea689a199e34a3be9a2c22d72adc9.jpg" \
    --output results/dashboard_tests

# Create test session directory with timestamp
SESSION_DIR="results/dashboard_tests/test_session_$(date +%Y%m%d_%H%M%S)"
echo "📁 Creating test session directory: $SESSION_DIR"
mkdir -p "$SESSION_DIR"

# Move all test directories into the session directory
echo "🔄 Organizing test results..."
find results/dashboard_tests -maxdepth 1 -type d -name "*patch*" -exec mv {} "$SESSION_DIR/" \;

# Update image paths in JSON files
echo "🖼️  Updating image paths..."
find "$SESSION_DIR" -name "test_report.json" -exec sed -i '' "s|results/dashboard_tests/|$SESSION_DIR/|g" {} \;

echo "✅ Test results organized successfully!"
echo "📂 Results location: $SESSION_DIR"

# Start dashboard
echo "🚀 Starting dashboard..."
cargo run --bin align dashboard --results-dir results/dashboard_tests
