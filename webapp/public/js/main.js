document.addEventListener('DOMContentLoaded', function() {
    // Global variables to store milestone data
    let allMilestones = [];
    let domainOptions = new Set();
    let domainToAgeRanges = {};
    let domainAgeToMilestones = {};
    
    // Add a direct milestone count update function
    function updateMilestoneCount(count, domains) {
        console.log(`Updating milestone count: ${count} milestones across ${domains.length} domains`);
        const milestoneCountElement = document.getElementById('milestone-count');
        if (milestoneCountElement) {
            console.log('Found milestone-count element, updating it');
            milestoneCountElement.textContent = `${count} milestones available across ${domains.length} domains: ${domains.join(', ')}`;
            milestoneCountElement.style.display = 'block';
        } else {
            console.error('Could not find milestone-count element');
        }
    }
    
    // Fetch milestone data when the page loads
    console.log('DOM content loaded, fetching milestones...');
    fetchMilestones();
    
    // Function to fetch all milestones from the API
    function fetchMilestones() {
        console.log('Fetching milestones from /milestones');
        
        // First try to fetch from our test endpoint
        fetch('/test-milestones')
            .then(response => {
                console.log('Test endpoint response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('Test endpoint response:', data);
                if (data.count) {
                    console.log(`Test endpoint reports ${data.count} milestones across domains: ${data.domains.join(', ')}`);
                    updateMilestoneCount(data.count, data.domains);
                }
            })
            .catch(error => {
                console.error('Error fetching from test endpoint:', error);
            });
        
        // Then fetch the actual milestones
        fetch('/milestones')
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log(`Loaded ${data.length} milestones from server`);
                console.log('First few milestones:', data.slice(0, 3));
                const domains = [...new Set(data.map(m => m.domain))];
                console.log('Domains:', domains);
                
                // Update the milestone count display
                updateMilestoneCount(data.length, domains);
                
                allMilestones = data;
                processMilestoneData(data);
                populateDomainDropdown();
            })
            .catch(error => {
                console.error('Error fetching milestones:', error);
                
                // Update the milestone count display with the error
                const milestoneCountElement = document.getElementById('milestone-count');
                if (milestoneCountElement) {
                    console.log('Found milestone-count element, updating with error');
                    milestoneCountElement.className = 'alert alert-danger mb-3';
                    milestoneCountElement.textContent = `Failed to load milestones. Error: ${error.message}`;
                } else {
                    console.error('Could not find milestone-count element to display error');
                }
                
                document.getElementById('milestone-select-container').innerHTML = 
                    `<div class="alert alert-danger">Failed to load milestones. Please try again later.<br>Error: ${error.message}</div>`;
            });
    }
    
    // Process milestone data to organize by domain and age range
    function processMilestoneData(milestones) {
        console.log('Processing milestone data...');
        milestones.forEach(milestone => {
            // Extract domain
            const domain = milestone.domain;
            domainOptions.add(domain);
            
            // Extract age range
            const ageRange = milestone.age_range;
            
            // Create domain to age ranges mapping
            if (!domainToAgeRanges[domain]) {
                domainToAgeRanges[domain] = new Set();
            }
            domainToAgeRanges[domain].add(ageRange);
            
            // Create domain+age to milestones mapping
            const key = `${domain}_${ageRange}`;
            if (!domainAgeToMilestones[key]) {
                domainAgeToMilestones[key] = [];
            }
            domainAgeToMilestones[key].push(milestone);
        });
        
        // Add milestone count to show the user how many options are available
        console.log('Domain options:', [...domainOptions]);
        console.log('Domain to age ranges:', domainToAgeRanges);
        
        const countContainer = document.createElement('div');
        countContainer.classList.add('text-muted', 'mb-3');
        countContainer.textContent = `${milestones.length} milestones available across ${domainOptions.size} domains`;
        console.log('Created count container with text:', countContainer.textContent);
        
        const containerElement = document.getElementById('milestone-select-container');
        if (containerElement) {
            console.log('Found milestone-select-container, inserting count');
            containerElement.insertBefore(countContainer, containerElement.firstChild);
        } else {
            console.error('Could not find milestone-select-container element');
        }
    }
    
    // Populate the domain dropdown
    function populateDomainDropdown() {
        console.log('Populating domain dropdown...');
        const domainSelect = document.getElementById('domain-select');
        if (!domainSelect) {
            console.error('Could not find domain-select element');
            return;
        }
        
        // Add default option
        domainSelect.innerHTML = '<option value="">Select Domain</option>';
        
        // Add domain options sorted alphabetically with descriptions
        Array.from(domainOptions).sort().forEach(domain => {
            const domainDisplayName = getDomainDisplayName(domain);
            const milestonesInDomain = allMilestones.filter(m => m.domain === domain).length;
            domainSelect.innerHTML += `<option value="${domain}">${domainDisplayName} (${milestonesInDomain})</option>`;
        });
        console.log('Domain dropdown populated with options:', domainSelect.innerHTML);
        
        // Add event listener for domain change
        domainSelect.addEventListener('change', function() {
            populateAgeRangeDropdown(this.value);
        });
    }
    
    // Return a human-readable name for each domain code
    function getDomainDisplayName(domainCode) {
        const domainMap = {
            'SOC': 'Social',
            'GM': 'Gross Motor',
            'FM': 'Fine Motor',
            'EL': 'Expressive Language',
            'RL': 'Receptive Language',
            'Cog': 'Cognitive',
            'Emo': 'Emotional',
            'ADL': 'Activities of Daily Living'
        };
        
        return domainMap[domainCode] || domainCode;
    }
    
    // Populate the age range dropdown based on selected domain
    function populateAgeRangeDropdown(domain) {
        const ageRangeSelect = document.getElementById('age-range-select');
        const milestoneSelect = document.getElementById('milestone-select');
        
        // Reset milestone dropdown
        milestoneSelect.innerHTML = '<option value="">Select Milestone</option>';
        milestoneSelect.disabled = true;
        
        if (!domain) {
            // If no domain selected, disable age range dropdown
            ageRangeSelect.innerHTML = '<option value="">Select Age Range</option>';
            ageRangeSelect.disabled = true;
            return;
        }
        
        // Enable age range dropdown
        ageRangeSelect.disabled = false;
        ageRangeSelect.innerHTML = '<option value="">Select Age Range</option>';
        
        // Add age range options sorted numerically
        const ageRanges = Array.from(domainToAgeRanges[domain] || []);
        ageRanges.sort((a, b) => {
            // Sort age ranges like "0-6", "6-12", etc.
            const aStart = parseInt(a.split('-')[0]);
            const bStart = parseInt(b.split('-')[0]);
            return aStart - bStart;
        }).forEach(ageRange => {
            const milestonesInAgeRange = domainAgeToMilestones[`${domain}_${ageRange}`].length;
            ageRangeSelect.innerHTML += `<option value="${ageRange}">${ageRange} months (${milestonesInAgeRange})</option>`;
        });
        
        // Add event listener for age range change
        ageRangeSelect.addEventListener('change', function() {
            populateMilestoneDropdown(domain, this.value);
        });
    }
    
    // Populate the milestone dropdown based on selected domain and age range
    function populateMilestoneDropdown(domain, ageRange) {
        const milestoneSelect = document.getElementById('milestone-select');
        
        if (!domain || !ageRange) {
            // If no domain or age range selected, disable milestone dropdown
            milestoneSelect.innerHTML = '<option value="">Select Milestone</option>';
            milestoneSelect.disabled = true;
            return;
        }
        
        // Enable milestone dropdown
        milestoneSelect.disabled = false;
        milestoneSelect.innerHTML = '<option value="">Select Milestone</option>';
        
        // Add milestone options
        const key = `${domain}_${ageRange}`;
        const milestones = domainAgeToMilestones[key] || [];
        milestones.sort((a, b) => a.behavior.localeCompare(b.behavior)).forEach(milestone => {
            milestoneSelect.innerHTML += `
                <option value="${milestone.behavior}" title="${milestone.criteria}">
                    ${milestone.behavior}
                </option>`;
        });
        
        // Add change event to update the criteria display
        milestoneSelect.addEventListener('change', function() {
            displayMilestoneCriteria(domain, ageRange, this.value);
        });
    }
    
    // Display milestone criteria when a milestone is selected
    function displayMilestoneCriteria(domain, ageRange, milestoneBehavior) {
        const criteriaContainer = document.getElementById('milestone-criteria');
        if (!criteriaContainer) return;
        
        if (!domain || !ageRange || !milestoneBehavior) {
            criteriaContainer.innerHTML = '';
            criteriaContainer.style.display = 'none';
            return;
        }
        
        const key = `${domain}_${ageRange}`;
        const milestones = domainAgeToMilestones[key] || [];
        const selectedMilestone = milestones.find(m => m.behavior === milestoneBehavior);
        
        if (selectedMilestone) {
            criteriaContainer.innerHTML = `
                <div class="alert alert-info">
                    <strong>Criteria:</strong> ${selectedMilestone.criteria}
                </div>`;
            criteriaContainer.style.display = 'block';
            
            // Pre-fill the question field with a standard question format
            const questionField = document.getElementById('question');
            if (questionField) {
                questionField.value = `Does your child ${milestoneBehavior.toLowerCase()}?`;
            }
        } else {
            criteriaContainer.innerHTML = '';
            criteriaContainer.style.display = 'none';
        }
    }
    
    // Handle comprehensive assessment form submission
    const comprehensiveForm = document.getElementById('comprehensive-form');
    const comprehensiveResult = document.getElementById('comprehensive-result');

    if (comprehensiveForm) {
        comprehensiveForm.addEventListener('submit', function(e) {
            console.log('Form submission intercepted');
            e.preventDefault(); // Prevent the default form submission
            console.log('Default form submission prevented');
            
            // Get form values
            const question = document.getElementById('question').value;
            const milestone = document.getElementById('milestone-select').value;
            const response = document.getElementById('response').value;
            const includeKeywords = document.getElementById('include-keywords').checked;
            if (includeKeywords) {
                // Add keyword data if the checkbox is checked
                requestData.keywords = {
                    "INDEPENDENT": getKeywordArray('independent-keywords'),
                    "EMERGING": getKeywordArray('emerging-keywords'),
                    "WITH_SUPPORT": getKeywordArray('support-keywords'),
                    "CANNOT_DO": getKeywordArray('cannot-keywords'),
                    "LOST_SKILL": getKeywordArray('lost-keywords')
                };
            }
            
            // Log current values for debugging
            console.log('Form values:', {
                question,
                milestone,
                response,
                includeKeywords
            });
            
            // Validate required fields
            if (!question || !milestone || !response) {
                comprehensiveResult.textContent = 'Error: Please fill in all required fields';
                console.error('Validation failed: Missing required fields');
                return;
            }
            
            // Prepare request data
            const requestData = {
                question: question,
                milestone_behavior: milestone,
                parent_response: response
            };
            
            // Use the direct-test endpoint which now has normalization logic
            const DIRECT_TEST_URL = '/direct-test';
            
            // Log more details about the milestone selection for debugging
            console.log('Selected milestone details:');
            console.log('Domain:', document.getElementById('domain-select').value);
            console.log('Age range:', document.getElementById('age-range-select').value);
            console.log('Milestone full name:', milestone);
            
            // Try to fetch the exact milestone by ID first, then fall back to fuzzy matching
            const key = `${document.getElementById('domain-select').value}_${document.getElementById('age-range-select').value}`;
            const milestones = domainAgeToMilestones[key] || [];
            const exactMilestone = milestones.find(m => m.behavior === milestone);
            
            console.log('Found exact milestone in frontend data:', exactMilestone ? 'Yes' : 'No');
            if (exactMilestone) {
                console.log('Full milestone data:', exactMilestone);
            }
            
            // Show loading state
            comprehensiveResult.textContent = 'Loading...';
            
            // Send request to the direct-test endpoint with normalization logic
            console.log('Sending request to:', DIRECT_TEST_URL);
            fetch(DIRECT_TEST_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => {
                console.log('Response received, status:', response.status);
                if (!response.ok) {
                    // Handle non-200 responses
                    return response.text().then(text => {
                        console.error('Error response text:', text);
                        try {
                            // Try to parse as JSON
                            const error = JSON.parse(text);
                            throw new Error(`API Error (${response.status}): ${error.detail || error.message || text}`);
                        } catch (e) {
                            // If not JSON, return the raw text
                            throw new Error(`API Error (${response.status}): ${text || 'Unknown error'}`);
                        }
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Success response data:', data);
                // Check if the score is NOT_RATED
                if (data.score_label === "NOT_RATED") {
                    // Display a more user-friendly message
                    comprehensiveResult.textContent = 'The system could not confidently determine a score based on the provided response. Please try providing more specific details about the child\'s behavior or add more relevant keywords.';
                } else {
                    // Display formatted result
                    comprehensiveResult.textContent = JSON.stringify(data, null, 2);
                }
            })
            .catch(error => {
                console.error('Fetch error:', error);
                comprehensiveResult.textContent = 'Error: ' + error.message;
            });
        });
    }
    
    // Handle health check button
    const checkHealthBtn = document.getElementById('check-health');
    const healthResult = document.getElementById('health-result');
    
    if (checkHealthBtn && healthResult) {
        checkHealthBtn.addEventListener('click', function() {
            // Show loading state
            healthResult.textContent = 'Checking API health...';
            
            // Send request to API
            fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                // Display formatted result
                healthResult.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                healthResult.textContent = 'Error: ' + error.message;
            });
        });
    }

    // Function to update the dropdown with milestone options for the selected domain and age range
    function updateMilestoneDropdown() {
        const domain = document.getElementById('domain-select').value;
        const ageRange = document.getElementById('age-range-select').value;
        const milestoneSelect = document.getElementById('milestone-select');
        const key = `${domain}_${ageRange}`;
        
        // Clear existing options
        milestoneSelect.innerHTML = '<option value="">Select a milestone</option>';
        
        // Update options based on selected domain and age range
        if (domainAgeToMilestones[key]) {
            const milestones = domainAgeToMilestones[key];
            milestones.forEach(milestone => {
                const option = document.createElement('option');
                option.value = milestone.behavior;
                option.textContent = milestone.behavior;
                milestoneSelect.appendChild(option);
            });
        }
    }

    // Helper function to convert keyword input to array
    function getKeywordArray(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return [];
        
        const keywords = element.value || '';
        return keywords.split(',')
            .map(k => k.trim())
            .filter(k => k.length > 0);
    }

    // Initialize domain and age range dropdowns with available options
}); 