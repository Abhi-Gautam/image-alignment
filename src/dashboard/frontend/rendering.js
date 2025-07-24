// Rendering functions for UI components

function renderTestSessions() {
    const sessionsList = document.getElementById('sessionsList');
    
    if (!filteredData.test_sessions.length) {
        sessionsList.innerHTML = '<div class="empty-state">No test sessions found</div>';
        return;
    }

    sessionsList.innerHTML = filteredData.test_sessions.map(session => `
        <div class="session-item ${selectedSession?.id === session.id ? 'active' : ''}" 
             onclick="selectSession('${session.id}')">
            <div class="session-info">
                <span class="session-title">${session.name}</span>
                <span class="session-meta">${new Date(session.created_at).toLocaleDateString()}</span>
            </div>
            <div class="session-meta">${session.sem_image_path.split('/').pop()}</div>
            <div class="session-stats">
                <span class="session-stat">
                    <span>Tests:</span>
                    <strong>${session.total_tests}</strong>
                </span>
                <span class="session-stat">
                    <span>Success:</span>
                    <strong class="${session.success_rate > 70 ? 'text-success' : 'text-error'}">
                        ${session.success_rate.toFixed(1)}%
                    </strong>
                </span>
                <span class="session-stat">
                    <span>Avg:</span>
                    <strong>${session.avg_processing_time.toFixed(1)}ms</strong>
                </span>
            </div>
        </div>
    `).join('');
}

function renderTestCases() {
    const testCasesList = document.getElementById('testCasesList');
    
    if (!selectedSession) {
        testCasesList.innerHTML = '<div class="empty-state">Select a test session to view details</div>';
        return;
    }

    if (!selectedSession.test_results.length) {
        testCasesList.innerHTML = '<div class="empty-state">No test cases found in this session</div>';
        return;
    }

    testCasesList.innerHTML = `
        <div class="test-grid">
            ${selectedSession.test_results.map(test => `
                <div class="test-item ${selectedTest?.test_id === test.test_id ? 'selected' : ''}"
                     onclick="selectTest('${test.test_id}')">
                    <div class="test-header">
                        <span class="test-name">${test.algorithm_name}</span>
                        <span class="test-status ${test.performance_metrics.success ? 'success' : 'failed'}">
                            ${test.performance_metrics.success ? '✓' : '✗'}
                        </span>
                    </div>
                    <div class="test-metrics">
                        <span>${test.patch_info.size[0]}×${test.patch_info.size[1]}</span>
                        <span>${test.transformation_applied.noise_parameters}</span>
                        <span>${test.performance_metrics.translation_error_px.toFixed(1)}px</span>
                        <span>${test.alignment_result.execution_time_ms.toFixed(1)}ms</span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function showImageDisplay() {
    if (!selectedTest) return;

    const imageResults = document.getElementById('imageResults');
    
    imageResults.innerHTML = `
        <div class="visual-results-layout">
            <!-- Row 1: Main SEM Image and Patches -->
            <div class="images-row">
                <div class="main-image-container">
                    <h4>SEM Image with Alignment Overlay</h4>
                    <img src="/api/image/${selectedTest.visual_outputs.overlay_result_path}" 
                         alt="Alignment Overlay" 
                         onclick="openModal(this)"
                         onerror="this.style.display='none'">
                </div>
                <div class="patches-container">
                    <div class="patch-image">
                        <h4>Original Patch</h4>
                        <img src="/api/image/${selectedTest.patch_info.patch_path}" 
                             alt="Original Patch" 
                             onclick="openModal(this)"
                             onerror="this.style.display='none'">
                    </div>
                    <div class="patch-image">
                        <h4>Transformed Patch</h4>
                        <img src="/api/image/${selectedTest.transformation_applied.transformed_patch_path}" 
                             alt="Transformed Patch" 
                             onclick="openModal(this)"
                             onerror="this.style.display='none'">
                    </div>
                </div>
            </div>
            
            <!-- Row 2: Metrics Data -->
            <div class="metrics-row">
                <div class="metrics-panel">
                    <h4 style="margin-bottom: var(--spacing-sm); color: var(--text-primary);">Performance Metrics</h4>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <span class="metric-label">Algorithm</span>
                            <span class="metric-value">${selectedTest.algorithm_name}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Translation Error</span>
                            <span class="metric-value">${selectedTest.performance_metrics.translation_error_px.toFixed(2)}px</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Confidence</span>
                            <span class="metric-value">${selectedTest.alignment_result.confidence.toFixed(3)}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Processing Time</span>
                            <span class="metric-value">${selectedTest.alignment_result.execution_time_ms.toFixed(1)}ms</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Status</span>
                            <span class="metric-value ${selectedTest.performance_metrics.success ? 'success' : 'failed'}">
                                ${selectedTest.performance_metrics.success ? '✓ Success' : '✗ Failed'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function clearImageDisplay() {
    const imageResults = document.getElementById('imageResults');
    imageResults.innerHTML = '<div class="empty-state">Select a test case to view visual results</div>';
}

function renderAlgorithmSummary(summaries) {
    const summaryContainer = document.getElementById('algorithmSummary');
    
    if (!summaries.length) {
        summaryContainer.innerHTML = '<div class="empty-state">Select a test session to view algorithm performance</div>';
        return;
    }

    // Sort by success rate descending
    summaries.sort((a, b) => b.success_rate - a.success_rate);

    summaryContainer.innerHTML = summaries.map(summary => `
        <div class="algorithm-card">
            <h4 class="algorithm-name">${summary.name}</h4>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${summary.success_rate}%"></div>
            </div>
            <div class="algorithm-stats">
                <div>
                    <span class="metric-label">Success Rate</span>
                    <strong>${summary.success_rate.toFixed(1)}%</strong>
                </div>
                <div>
                    <span class="metric-label">Total Tests</span>
                    <strong>${summary.total_tests}</strong>
                </div>
                <div>
                    <span class="metric-label">Avg Error</span>
                    <strong>${summary.avg_translation_error.toFixed(1)}px</strong>
                </div>
                <div>
                    <span class="metric-label">Avg Time</span>
                    <strong>${summary.avg_processing_time.toFixed(1)}ms</strong>
                </div>
            </div>
        </div>
    `).join('');
}