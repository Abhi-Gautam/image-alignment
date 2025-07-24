// Global state
let dashboardData = null;
let filteredData = null;
let selectedSession = null;
let selectedTest = null;
let currentFilters = {
    algorithm: '',
    patchSize: '',
    transformation: '',
    successOnly: false
};

// Theme management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    document.getElementById('themeIcon').textContent = theme === 'light' ? '🌙' : '☀️';
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    loadDashboardData();
    setupEventListeners();
});

function setupEventListeners() {
    // Store filter values before they change
    const algorithmFilter = document.getElementById('algorithmFilter');
    const patchSizeFilter = document.getElementById('patchSizeFilter');
    const transformationFilter = document.getElementById('transformationFilter');
    const successOnlyFilter = document.getElementById('successOnlyFilter');

    algorithmFilter.addEventListener('change', (e) => {
        currentFilters.algorithm = e.target.value;
        applyFilters();
    });

    patchSizeFilter.addEventListener('change', (e) => {
        currentFilters.patchSize = e.target.value;
        applyFilters();
    });

    transformationFilter.addEventListener('change', (e) => {
        currentFilters.transformation = e.target.value;
        applyFilters();
    });

    successOnlyFilter.addEventListener('change', (e) => {
        currentFilters.successOnly = e.target.checked;
        applyFilters();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.querySelector('.modal')) {
            closeModal();
        }
    });
}

async function loadDashboardData() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) throw new Error('Failed to fetch data');
        
        dashboardData = await response.json();
        filteredData = dashboardData;
        updateDashboard();
        updateAlgorithmSummary();
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showToast('Failed to load dashboard data', 'error');
    }
}

function calculateSessionAlgorithmSummary(session) {
    if (!session || !session.test_results.length) {
        return [];
    }

    const algorithmStats = {};
    
    // Group test results by algorithm
    session.test_results.forEach(test => {
        if (!algorithmStats[test.algorithm_name]) {
            algorithmStats[test.algorithm_name] = {
                name: test.algorithm_name,
                total_tests: 0,
                success_count: 0,
                translation_errors: [],
                processing_times: [],
                confidences: []
            };
        }
        
        const stats = algorithmStats[test.algorithm_name];
        stats.total_tests++;
        
        if (test.performance_metrics.success) {
            stats.success_count++;
        }
        
        stats.translation_errors.push(test.performance_metrics.translation_error_px);
        stats.processing_times.push(test.alignment_result.execution_time_ms);
        stats.confidences.push(test.alignment_result.confidence);
    });
    
    // Calculate summary metrics
    return Object.values(algorithmStats).map(stats => ({
        name: stats.name,
        total_tests: stats.total_tests,
        success_count: stats.success_count,
        success_rate: stats.total_tests > 0 ? (stats.success_count / stats.total_tests) * 100 : 0,
        avg_translation_error: stats.translation_errors.reduce((a, b) => a + b, 0) / stats.translation_errors.length,
        avg_processing_time: stats.processing_times.reduce((a, b) => a + b, 0) / stats.processing_times.length,
        avg_confidence: stats.confidences.reduce((a, b) => a + b, 0) / stats.confidences.length
    }));
}

function updateAlgorithmSummary() {
    if (!selectedSession) {
        renderAlgorithmSummary([]);
        return;
    }
    
    const summaries = calculateSessionAlgorithmSummary(selectedSession);
    renderAlgorithmSummary(summaries);
}

function updateDashboard() {
    if (!filteredData) return;

    // Update global stats
    const totalTests = filteredData.test_sessions.reduce((sum, session) => sum + session.total_tests, 0);
    const totalSuccessful = filteredData.test_sessions.reduce((sum, session) => 
        sum + session.test_results.filter(t => t.performance_metrics.success).length, 0);
    const avgSuccessRate = totalTests > 0 ? (totalSuccessful / totalTests * 100) : 0;
    const avgTime = filteredData.test_sessions.reduce((sum, session) => sum + session.avg_processing_time, 0) / 
                    (filteredData.test_sessions.length || 1);

    document.getElementById('totalTests').textContent = totalTests.toLocaleString();
    document.getElementById('successRate').textContent = avgSuccessRate.toFixed(1) + '%';
    document.getElementById('avgTime').textContent = avgTime.toFixed(1);
    document.getElementById('testSessions').textContent = filteredData.test_sessions.length;

    // Populate filter options from original data
    populateFilterOptions();
    
    // Restore filter values
    document.getElementById('algorithmFilter').value = currentFilters.algorithm;
    document.getElementById('patchSizeFilter').value = currentFilters.patchSize;
    document.getElementById('transformationFilter').value = currentFilters.transformation;
    document.getElementById('successOnlyFilter').checked = currentFilters.successOnly;
    
    // Render sessions
    renderTestSessions();
    updateFilterStatus();
}

function populateFilterOptions() {
    if (!dashboardData) return;

    const algorithmSelect = document.getElementById('algorithmFilter');
    const patchSizeSelect = document.getElementById('patchSizeFilter');
    const transformationSelect = document.getElementById('transformationFilter');

    // Clear existing options except the first one
    algorithmSelect.innerHTML = '<option value="">All Algorithms</option>';
    patchSizeSelect.innerHTML = '<option value="">All Sizes</option>';
    transformationSelect.innerHTML = '<option value="">All Transformations</option>';

    // Add options from original data (not filtered data)
    dashboardData.algorithms.forEach(algorithm => {
        const option = document.createElement('option');
        option.value = algorithm;
        option.textContent = algorithm;
        algorithmSelect.appendChild(option);
    });

    dashboardData.patch_sizes.forEach(size => {
        const option = document.createElement('option');
        option.value = size;
        option.textContent = size;
        patchSizeSelect.appendChild(option);
    });

    dashboardData.transformations.forEach(transformation => {
        const option = document.createElement('option');
        option.value = transformation;
        option.textContent = transformation.replace(/_/g, ' ');
        transformationSelect.appendChild(option);
    });
}