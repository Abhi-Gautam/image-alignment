// Interaction handlers and event functions

let currentMode = 'test-runner'; // 'filter' or 'test-runner'

function switchToFilterMode() {
    currentMode = 'filter';
    document.getElementById('filterModeBtn').classList.add('active');
    document.getElementById('testRunnerModeBtn').classList.remove('active');
    document.getElementById('filterActions').style.display = 'flex';
    document.getElementById('testRunnerSection').style.display = 'none';
    document.querySelector('.filter-grid').style.display = 'grid';
}

function switchToTestRunnerMode() {
    currentMode = 'test-runner';
    document.getElementById('testRunnerModeBtn').classList.add('active');
    document.getElementById('filterModeBtn').classList.remove('active');
    document.getElementById('filterActions').style.display = 'none';
    document.getElementById('testRunnerSection').style.display = 'block';
    document.querySelector('.filter-grid').style.display = 'none';
}

function selectSession(sessionId) {
    selectedSession = filteredData.test_sessions.find(s => s.id === sessionId);
    selectedTest = null;
    renderTestSessions();
    renderTestCases();
    clearImageDisplay();
    updateAlgorithmSummary();
}

function selectTest(testId) {
    selectedTest = selectedSession.test_results.find(t => t.test_id === testId);
    renderTestCases();
    showImageDisplay();
}

async function applyFilters() {
    const params = new URLSearchParams();
    
    // Add multi-select filter values
    if (selectedFilterAlgorithms.length > 0) {
        params.append('algorithms', selectedFilterAlgorithms.join(','));
    }
    if (selectedFilterPatchSizes.length > 0) {
        params.append('patch_sizes', selectedFilterPatchSizes.join(','));
    }
    if (selectedFilterTransformations.length > 0) {
        params.append('transformations', selectedFilterTransformations.join(','));
    }
    if (currentFilters.successOnly) params.append('success_only', 'true');

    try {
        const response = await fetch(`/api/data?${params}`);
        if (!response.ok) throw new Error('Failed to fetch filtered data');
        
        filteredData = await response.json();
        
        // Check if selected session still exists in filtered data
        if (selectedSession) {
            const updatedSession = filteredData.test_sessions.find(s => s.id === selectedSession.id);
            if (updatedSession && updatedSession.test_results.length > 0) {
                selectedSession = updatedSession;
                if (selectedTest) {
                    const testStillExists = updatedSession.test_results.find(t => t.test_id === selectedTest.test_id);
                    if (!testStillExists) {
                        selectedTest = null;
                        clearImageDisplay();
                    }
                }
            } else {
                selectedSession = null;
                selectedTest = null;
                clearImageDisplay();
            }
        }
        
        updateDashboard();
        if (selectedSession) {
            renderTestCases();
        }
        updateFilterStatus();
    } catch (error) {
        console.error('Error applying filters:', error);
        showToast('Failed to apply filters', 'error');
    }
}

function updateFilterStatus() {
    const hasActiveFilters = selectedFilterAlgorithms.length > 0 || 
                           selectedFilterPatchSizes.length > 0 || 
                           selectedFilterTransformations.length > 0 || 
                           currentFilters.successOnly;
    
    const filterStatus = document.getElementById('filterStatus');
    filterStatus.textContent = hasActiveFilters ? 'Active' : '';
}

function clearFilters() {
    currentFilters = {
        successOnly: false
    };
    
    selectedFilterAlgorithms = [];
    selectedFilterPatchSizes = [];
    selectedFilterTransformations = [];
    
    // Clear all checkboxes in filter dropdowns
    document.querySelectorAll('#filterAlgorithmsDropdown input[type="checkbox"]').forEach(cb => cb.checked = false);
    document.querySelectorAll('#filterPatchSizesDropdown input[type="checkbox"]').forEach(cb => cb.checked = false);
    document.querySelectorAll('#filterTransformationsDropdown input[type="checkbox"]').forEach(cb => cb.checked = false);
    document.getElementById('successOnlyFilter').checked = false;
    
    // Update display text (skip filters to avoid triggering during clear)
    updateMultiSelect('filterAlgorithms', true);
    updateMultiSelect('filterPatchSizes', true);
    updateMultiSelect('filterTransformations', true);
    
    applyFilters();
}

async function refreshData() {
    try {
        const response = await fetch('/api/refresh');
        if (!response.ok) throw new Error('Failed to refresh data');
        
        await loadDashboardData();
        showToast('Data refreshed successfully', 'success');
    } catch (error) {
        console.error('Error refreshing data:', error);
        showToast('Failed to refresh data', 'error');
    }
}

// Modal functionality
function openModal(img) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.onclick = closeModal;
    
    const modalImg = img.cloneNode();
    modalImg.style.cssText = '';
    modal.appendChild(modalImg);
    
    document.body.appendChild(modal);
}

function closeModal() {
    const modal = document.querySelector('.modal');
    if (modal) modal.remove();
}

// Toast notifications
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Multi-select dropdown functions
function toggleDropdown(type) {
    const dropdown = document.getElementById(type + 'Dropdown');
    const toggle = dropdown.previousElementSibling;
    
    if (dropdown.style.display === 'none') {
        dropdown.style.display = 'block';
        toggle.classList.add('open');
    } else {
        dropdown.style.display = 'none';
        toggle.classList.remove('open');
    }
}

function updateMultiSelect(type, skipFilters = false) {
    const dropdown = document.getElementById(type + 'Dropdown');
    const checkboxes = dropdown.querySelectorAll('input[type="checkbox"]:checked');
    const textElement = document.getElementById(type + 'Text');
    
    let defaultText = '';
    let selectedValues = [];
    
    // Determine default text and update appropriate array
    switch(type) {
        case 'patchSizes':
            defaultText = 'Select patch sizes...';
            break;
        case 'scenarios':
            defaultText = 'Select scenarios...';
            break;
        case 'filterAlgorithms':
            defaultText = 'All algorithms...';
            selectedFilterAlgorithms = Array.from(checkboxes).map(cb => cb.value);
            break;
        case 'filterPatchSizes':
            defaultText = 'All sizes...';
            selectedFilterPatchSizes = Array.from(checkboxes).map(cb => cb.value);
            break;
        case 'filterTransformations':
            defaultText = 'All transformations...';
            selectedFilterTransformations = Array.from(checkboxes).map(cb => cb.value);
            break;
    }
    
    if (checkboxes.length === 0) {
        textElement.textContent = defaultText;
    } else {
        const values = Array.from(checkboxes).map(cb => {
            const label = cb.parentElement.textContent.trim();
            return label;
        });
        textElement.textContent = values.join(', ');
    }
    
    // Apply filters if it's a filter dropdown and not during initialization
    if (type.startsWith('filter') && !skipFilters) {
        applyFilters();
    }
}

// Close dropdowns when clicking outside
document.addEventListener('click', function(event) {
    // Don't close dropdown if clicking on checkbox or label inside dropdown
    if (!event.target.closest('.multi-select-dropdown') && 
        !event.target.matches('input[type="checkbox"]') && 
        !event.target.closest('label')) {
        document.querySelectorAll('.multi-select-options').forEach(dropdown => {
            dropdown.style.display = 'none';
            dropdown.previousElementSibling.classList.remove('open');
        });
    }
});

// Initialize selected values on page load
document.addEventListener('DOMContentLoaded', function() {
    updateMultiSelect('patchSizes');
    updateMultiSelect('scenarios');
});

// Test Runner functionality
async function runVisualTest() {
    const semImageInput = document.getElementById('testSemImage');
    const patchSizesDropdown = document.getElementById('patchSizesDropdown');
    const scenariosDropdown = document.getElementById('scenariosDropdown');
    
    // Validate inputs
    if (!semImageInput.files[0]) {
        showToast('Please select a SEM image file', 'error');
        return;
    }
    
    const selectedPatchSizes = Array.from(patchSizesDropdown.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
    const selectedScenarios = Array.from(scenariosDropdown.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
    
    if (selectedPatchSizes.length === 0) {
        showToast('Please select at least one patch size', 'error');
        return;
    }
    
    if (selectedScenarios.length === 0) {
        showToast('Please select at least one test scenario', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('sem_image', semImageInput.files[0]);
    formData.append('patch_sizes', selectedPatchSizes.join(','));
    formData.append('scenarios', selectedScenarios.join(','));
    
    // Show progress and disable button
    const runBtn = document.getElementById('runTestBtn');
    const btnText = document.getElementById('runTestBtnText');
    const spinner = document.getElementById('runTestSpinner');
    const progress = document.getElementById('testProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    runBtn.disabled = true;
    btnText.textContent = 'Running Test...';
    spinner.style.display = 'block';
    progress.style.display = 'block';
    
    try {
        // Start the test
        const response = await fetch('/api/run-visual-test', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to start visual test');
        }
        
        const result = await response.json();
        
        if (result.status === 'started') {
            // Poll for progress
            await pollTestProgress(result.test_id, progressFill, progressText);
            
            // Show success and refresh data
            showToast('Visual test completed successfully!', 'success');
            await refreshData();
            
            // Switch back to filter mode to see results
            switchToFilterMode();
        }
        
    } catch (error) {
        console.error('Test failed:', error);
        showToast('Test failed: ' + error.message, 'error');
    } finally {
        // Reset UI
        runBtn.disabled = false;
        btnText.textContent = 'Run Visual Test';
        spinner.style.display = 'none';
        progress.style.display = 'none';
        progressFill.style.width = '0';
    }
}

async function pollTestProgress(testId, progressFill, progressText) {
    return new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/api/test-progress/${testId}`);
                if (!response.ok) {
                    throw new Error('Failed to get test progress');
                }
                
                const progress = await response.json();
                
                // Update progress bar
                progressFill.style.width = `${progress.percentage}%`;
                progressText.textContent = progress.message;
                
                if (progress.status === 'completed') {
                    clearInterval(interval);
                    resolve();
                } else if (progress.status === 'failed') {
                    clearInterval(interval);
                    reject(new Error(progress.error || 'Test failed'));
                }
                
            } catch (error) {
                clearInterval(interval);
                reject(error);
            }
        }, 1000); // Poll every second
    });
}

// Log viewer functionality
function openLogViewer(sessionId, testId) {
    // Create and show modal for log viewing
    createLogModal(sessionId, testId);
}

async function createLogModal(sessionId, testId) {
    // Create modal overlay
    const modalOverlay = document.createElement('div');
    modalOverlay.className = 'log-modal-overlay';
    modalOverlay.onclick = (e) => {
        if (e.target === modalOverlay) {
            closeLogModal();
        }
    };
    
    // Create modal content
    const modalContent = document.createElement('div');
    modalContent.className = 'log-modal-content';
    
    modalContent.innerHTML = `
        <div class="log-modal-header">
            <h3>Algorithm Execution Log</h3>
            <button class="close-btn" onclick="closeLogModal()">×</button>
        </div>
        <div class="log-modal-body">
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>Loading algorithm logs...</p>
            </div>
        </div>
    `;
    
    modalOverlay.appendChild(modalContent);
    document.body.appendChild(modalOverlay);
    
    // Load log content
    try {
        const response = await fetch(`/api/test-logs/${sessionId}/${testId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to load logs: ${response.status}`);
        }
        
        const logContent = await response.text();
        
        // Update modal with log content
        const modalBody = modalContent.querySelector('.log-modal-body');
        modalBody.innerHTML = `
            <div class="log-content">
                <pre>${logContent}</pre>
            </div>
            <div class="log-actions">
                <button class="btn btn-secondary" onclick="copyLogToClipboard()">
                    📋 Copy to Clipboard
                </button>
            </div>
        `;
        
    } catch (error) {
        console.error('Failed to load log:', error);
        const modalBody = modalContent.querySelector('.log-modal-body');
        modalBody.innerHTML = `
            <div class="error-message">
                <p>Failed to load algorithm logs: ${error.message}</p>
                <button class="btn btn-secondary" onclick="closeLogModal()">Close</button>
            </div>
        `;
    }
}

function closeLogModal() {
    const modal = document.querySelector('.log-modal-overlay');
    if (modal) {
        modal.remove();
    }
}

function copyLogToClipboard() {
    const logContent = document.querySelector('.log-content pre');
    if (logContent) {
        navigator.clipboard.writeText(logContent.textContent).then(() => {
            // Show brief success message
            const button = event.target;
            const originalText = button.textContent;
            button.textContent = '✓ Copied!';
            setTimeout(() => {
                button.textContent = originalText;
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy to clipboard:', err);
        });
    }
}