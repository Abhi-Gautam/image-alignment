// Interaction handlers and event functions

function selectSession(sessionId) {
    selectedSession = filteredData.test_sessions.find(s => s.id === sessionId);
    selectedTest = null;
    renderTestSessions();
    renderTestCases();
    clearImageDisplay();
}

function selectTest(testId) {
    selectedTest = selectedSession.test_results.find(t => t.test_id === testId);
    renderTestCases();
    showImageDisplay();
}

async function applyFilters() {
    const params = new URLSearchParams();
    if (currentFilters.algorithm) params.append('algorithm', currentFilters.algorithm);
    if (currentFilters.patchSize) params.append('patch_size', currentFilters.patchSize);
    if (currentFilters.transformation) params.append('transformation', currentFilters.transformation);
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
    } catch (error) {
        console.error('Error applying filters:', error);
        showToast('Failed to apply filters', 'error');
    }
}

function updateFilterStatus() {
    const hasActiveFilters = currentFilters.algorithm || 
                           currentFilters.patchSize || 
                           currentFilters.transformation || 
                           currentFilters.successOnly;
    
    const filterStatus = document.getElementById('filterStatus');
    filterStatus.textContent = hasActiveFilters ? 'Active' : '';
}

function clearFilters() {
    currentFilters = {
        algorithm: '',
        patchSize: '',
        transformation: '',
        successOnly: false
    };
    
    document.getElementById('algorithmFilter').value = '';
    document.getElementById('patchSizeFilter').value = '';
    document.getElementById('transformationFilter').value = '';
    document.getElementById('successOnlyFilter').checked = false;
    
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