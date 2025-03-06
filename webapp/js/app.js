/**
 * ASD Assessment API Demo Application
 * JavaScript for handling API interactions
 */

$(document).ready(function() {
    // API base URL - using proxy to avoid CORS issues
    const API_BASE_URL = '/api';
    
    // Common function to format JSON for display
    function formatJSON(json) {
        if (typeof json === 'string') {
            json = JSON.parse(json);
        }
        return JSON.stringify(json, null, 2);
    }
    
    // Common function to display error messages
    function showError(elementId, error) {
        $(`#${elementId}`).html(`<span class="error-text">Error: ${error.message || error}</span>`);
    }
    
    // Common function to handle API requests
    function makeApiRequest(endpoint, data, responseElementId, timeElementId) {
        // Clear previous response
        $(`#${responseElementId}`).html('<div class="loading"></div> Processing request...');
        $(`#${timeElementId}`).text('');
        
        // Record start time
        const startTime = performance.now();
        
        // Make API request
        $.ajax({
            url: `${API_BASE_URL}${endpoint}`,
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function(response) {
                // Calculate response time
                const endTime = performance.now();
                const responseTime = ((endTime - startTime) / 1000).toFixed(4);
                
                // Display response
                $(`#${responseElementId}`).html(formatJSON(response));
                $(`#${timeElementId}`).text(`${responseTime}s`);
            },
            error: function(xhr, status, error) {
                // Calculate response time even for errors
                const endTime = performance.now();
                const responseTime = ((endTime - startTime) / 1000).toFixed(4);
                
                $(`#${timeElementId}`).text(`${responseTime}s`);
                
                // Try to parse error response
                let errorMessage = error;
                try {
                    const errorResponse = JSON.parse(xhr.responseText);
                    errorMessage = errorResponse.detail || errorResponse.message || error;
                } catch (e) {
                    errorMessage = `${xhr.status}: ${error}`;
                }
                
                showError(responseElementId, errorMessage);
            }
        });
    }
    
    // Question Endpoint Handler
    $('#sendQuestionBtn').click(function() {
        const data = {
            text: $('#questionText').val(),
            milestone_id: $('#milestoneId').val()
        };
        
        makeApiRequest('/question', data, 'questionResponse', 'questionResponseTime');
    });
    
    // Keywords Endpoint Handler
    $('#sendKeywordsBtn').click(function() {
        // Parse the comma-separated keywords
        const keywordsText = $('#keywordsList').val();
        const keywords = keywordsText.split(',').map(k => k.trim()).filter(k => k);
        
        const data = {
            category: $('#keywordCategory').val(),
            keywords: keywords
        };
        
        makeApiRequest('/keywords', data, 'keywordsResponse', 'keywordsResponseTime');
    });
    
    // Send Score Endpoint Handler
    $('#sendScoreBtn').click(function() {
        const data = {
            milestone_id: $('#scoreId').val(),
            score: parseInt($('#scoreValue').val()),
            score_label: $('#scoreLabel').val()
        };
        
        makeApiRequest('/send-score', data, 'scoreResponse', 'scoreResponseTime');
    });
    
    // Score Response Endpoint Handler
    $('#sendResponseBtn').click(function() {
        const data = {
            milestone_behavior: $('#milestoneBehavior').val(),
            response: $('#parentResponse').val()
        };
        
        makeApiRequest('/score-response', data, 'responseResponse', 'responseResponseTime');
    });
    
    // Add visual feedback when switching tabs
    $('#apiTabs button').on('shown.bs.tab', function (e) {
        // Add a subtle animation when switching tabs
        $($(e.target).data('bs-target')).addClass('animate__animated animate__fadeIn');
        
        // Remove animation class after animation completes
        setTimeout(function() {
            $('.tab-pane').removeClass('animate__animated animate__fadeIn');
        }, 500);
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
}); 