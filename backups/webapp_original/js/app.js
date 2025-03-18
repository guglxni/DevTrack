/**
 * ASD Assessment API Demo Application
 * JavaScript for handling API interactions
 */

// API endpoint configuration
const API_BASE_URL = '/api';

// Default keywords for reliable scoring - added for consistent results
const reliableKeywords = {
    "Recognizes familiar people": {
        "INDEPENDENT": [
            "always recognizes",
            "consistently recognizes", 
            "easily recognizes",
            "immediately recognizes",
            "recognizes instantly",
            "definitely recognizes",
            "clearly recognizes",
            "recognizes without issues",
            "knows family members",
            "recognizes everyone",
            "knows everyone",
            "distinguishes between strangers",
            "smiles at familiar people",
            "yes he recognizes",
            "yes she recognizes",
            "yes they recognize",
            "always smiles when he sees"
        ],
        "WITH_SUPPORT": [
            "recognizes with help", 
            "sometimes recognizes", 
            "recognizes when prompted",
            "recognizes with assistance",
            "recognizes with support",
            "recognizes with guidance",
            "recognizes when reminded"
        ],
        "EMERGING": [
            "starting to recognize", 
            "beginning to recognize",
            "occasionally recognizes",
            "recognizes inconsistently",
            "sometimes seems to recognize",
            "might recognize",
            "recognizes rarely"
        ],
        "LOST_SKILL": [
            "used to recognize",
            "previously recognized",
            "recognized before",
            "no longer recognizes",
            "stopped recognizing",
            "lost ability to recognize"
        ],
        "CANNOT_DO": [
            "doesn't recognize anyone",
            "does not recognize anyone",
            "unable to recognize",
            "never recognizes",
            "can't recognize",
            "cannot recognize anyone",
            "fails to recognize",
            "shows no recognition",
            "treats everyone as strangers",
            "doesn't know who people are"
        ]
    }
};

// Generic keywords for other milestones
const genericKeywords = {
    "INDEPENDENT": [
        "definitely",
        "always",
        "consistently",
        "very well",
        "yes",
        "without any issues",
        "independently",
        "has mastered",
        "regularly"
    ],
    "WITH_SUPPORT": [
        "with help",
        "with assistance",
        "needs support",
        "with guidance",
        "when prompted",
        "when reminded",
        "needs help"
    ],
    "EMERGING": [
        "starting to",
        "beginning to",
        "occasionally",
        "sometimes",
        "inconsistently",
        "might",
        "trying to",
        "learning to"
    ],
    "LOST_SKILL": [
        "used to",
        "previously",
        "no longer",
        "stopped",
        "lost ability",
        "could before",
        "regressed"
    ],
    "CANNOT_DO": [
        "doesn't",
        "does not",
        "cannot",
        "never",
        "unable to",
        "fails to",
        "not able to",
        "hasn't",
        "has not"
    ]
};

// Helper function to get reliable keywords for a milestone
function getReliableKeywords(milestone) {
    if (reliableKeywords[milestone]) {
        return reliableKeywords[milestone];
    }
    return genericKeywords;
}

$(document).ready(function() {
    // Common function to format JSON for display
    function formatJSON(json) {
        if (typeof json === 'string') {
            json = JSON.parse(json);
        }
        return JSON.stringify(json, null, 2)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/("(\w+)":)/g, '<span class="json-key">$1</span>')
            .replace(/("([^"]*)":)/g, '<span class="json-key">$1</span>');
    }
    
    // Common function to display error messages
    function showError(elementId, error) {
        $(`#${elementId}`).html(`<div class="error">${error}</div>`);
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
                showError(responseElementId, `API Error: ${xhr.status} - ${error}`);
                $(`#${timeElementId}`).text('Failed');
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
    
    // Comprehensive Assessment Endpoint Handler
    $('#sendComprehensiveBtn').click(function() {
        // Get values from form
        const question = $('#compQuestion').val();
        const milestone = $('#compMilestoneBehavior').val();
        const response = $('#compParentResponse').val();
        const includeKeywords = $('#includeKeywords').is(':checked');
        
        // Build data object
        const data = {
            question: question,
            milestone_behavior: milestone,
            parent_response: response
        };
        
        // Add keywords if checkbox is checked
        if (includeKeywords) {
            // Build keywords object from inputs
            const keywords = {
                "CANNOT_DO": parseKeywords($('#keywordsCannot').val()),
                "LOST_SKILL": parseKeywords($('#keywordsLost').val()),
                "EMERGING": parseKeywords($('#keywordsEmerging').val()),
                "WITH_SUPPORT": parseKeywords($('#keywordsSupport').val()),
                "INDEPENDENT": parseKeywords($('#keywordsIndependent').val())
            };
            
            data.keywords = keywords;
        }
        
        // Make the API request
        makeApiRequest('/comprehensive-assessment', data, 'comprehensiveResponse', 'comprehensiveResponseTime');
    });
    
    // Helper function to parse comma-separated keywords
    function parseKeywords(input) {
        return input.split(',').map(k => k.trim()).filter(k => k);
    }
    
    // Toggle keywords section based on checkbox
    $('#includeKeywords').change(function() {
        if ($(this).is(':checked')) {
            $('#keywordsSection').slideDown();
        } else {
            $('#keywordsSection').slideUp();
        }
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