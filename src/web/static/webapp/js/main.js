document.addEventListener('DOMContentLoaded', function() {
    // Global variables to store milestone data
    let allMilestones = [];
    let domainOptions = new Set();
    let domainToAgeRanges = {};
    let domainAgeToMilestones = {};
    
    // Fetch milestone data when the page loads
    fetchMilestones();
    
    // Function to fetch all milestones from the API
    function fetchMilestones() {
        fetch('/milestones')
            .then(response => response.json())
            .then(data => {
                console.log(`Loaded ${data.length} milestones from server`);
                allMilestones = data;
                processMilestoneData(data);
                populateDomainDropdown();
            })
            .catch(error => {
                console.error('Error fetching milestones:', error);
                document.getElementById('milestone-select-container').innerHTML = 
                    `<div class="alert alert-danger">Failed to load milestones. Please try again later.<br>Error: ${error.message}</div>`;
            });
    }
    
    // Process milestone data to organize by domain and age range
    function processMilestoneData(milestones) {
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
        const countContainer = document.createElement('div');
        countContainer.classList.add('text-muted', 'mb-3');
        countContainer.textContent = `${milestones.length} milestones available across ${domainOptions.size} domains`;
        
        const containerElement = document.getElementById('milestone-select-container');
        if (containerElement) {
            containerElement.insertBefore(countContainer, containerElement.firstChild);
        }
    }
    
    // Populate the domain dropdown
    function populateDomainDropdown() {
        const domainSelect = document.getElementById('domain-select');
        if (!domainSelect) return;
        
        // Add default option
        domainSelect.innerHTML = '<option value="">Select Domain</option>';
        
        // Add domain options sorted alphabetically with descriptions
        Array.from(domainOptions).sort().forEach(domain => {
            const domainDisplayName = getDomainDisplayName(domain);
            const milestonesInDomain = allMilestones.filter(m => m.domain === domain).length;
            domainSelect.innerHTML += `<option value="${domain}">${domainDisplayName} (${milestonesInDomain})</option>`;
        });
        
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
    
    // Check API health on tab click
    document.getElementById('health-tab').addEventListener('click', checkAPIHealth);
    
    // Function to check the API health
    function checkAPIHealth() {
        const healthContainer = document.getElementById('health-container');
        healthContainer.innerHTML = `
            <p>Checking API health...</p>
            <div class="d-flex align-items-center">
                <div class="spinner-border text-primary me-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span>Connecting to API...</span>
            </div>
        `;
        
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                let statusClass = data.status === 'healthy' ? 'success' : 'warning';
                
                healthContainer.innerHTML = `
                    <div class="alert alert-${statusClass}">
                        <h5><i class="bi bi-check-circle-fill me-2"></i>API Status: ${data.status}</h5>
                        <p class="mb-0">Version: ${data.version}</p>
                    </div>
                    
                    <div class="card mt-3">
                        <div class="card-header">Endpoint Health Checks</div>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Scoring API
                                <span class="badge bg-success rounded-pill">Available</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Comprehensive Assessment
                                <span class="badge bg-success rounded-pill">Available</span>
                            </li>
                        </ul>
                    </div>
                `;
                
                // Check additional endpoints
                checkEndpoint('/r2r/health', 'R2R System');
                checkEndpoint('/active-learning/health', 'Active Learning');
            })
            .catch(error => {
                healthContainer.innerHTML = `
                    <div class="alert alert-danger">
                        <h5><i class="bi bi-exclamation-triangle-fill me-2"></i>API Connection Error</h5>
                        <p class="mb-0">Failed to connect to the API server. Please ensure the server is running.</p>
                        <small class="text-muted">${error.message}</small>
                    </div>
                `;
            });
    }
    
    // Check status of individual endpoints
    function checkEndpoint(url, name) {
        const healthList = document.querySelector('#health-container .list-group');
        if (!healthList) return;
        
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item d-flex justify-content-between align-items-center';
        listItem.innerHTML = `
            ${name}
            <span class="badge bg-secondary rounded-pill">Checking...</span>
        `;
        healthList.appendChild(listItem);
        
        fetch(url)
            .then(response => {
                const badge = listItem.querySelector('.badge');
                if (response.ok) {
                    badge.className = 'badge bg-success rounded-pill';
                    badge.textContent = 'Available';
                } else {
                    badge.className = 'badge bg-warning rounded-pill';
                    badge.textContent = `Error ${response.status}`;
                }
            })
            .catch(error => {
                const badge = listItem.querySelector('.badge');
                badge.className = 'badge bg-danger rounded-pill';
                badge.textContent = 'Unavailable';
            });
    }
    
    // Handle comprehensive assessment form submission
    document.getElementById('comprehensive-form').addEventListener('submit', function(event) {
        event.preventDefault();
        
        const questionInput = document.getElementById('question');
        const parentResponseInput = document.getElementById('parent-response');
        const milestoneSelect = document.getElementById('milestone-select');
        const submitBtn = document.getElementById('submit-btn');
        const resultContainer = document.getElementById('result-container');
        const resultJson = document.getElementById('result-json');
        
        // Basic validation
        if (!questionInput.value.trim()) {
            alert('Please enter a question.');
            questionInput.focus();
            return;
        }
        
        if (!parentResponseInput.value.trim()) {
            alert('Please enter the parent\'s response.');
            parentResponseInput.focus();
            return;
        }
        
        // Disable form and show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
        
        // Prepare request data
        const requestData = {
            question: questionInput.value.trim(),
            parent_response: parentResponseInput.value.trim()
        };
        
        // Add milestone if selected
        if (milestoneSelect.value) {
            const selectedMilestone = allMilestones.find(m => 
                m.behavior === milestoneSelect.value && 
                m.domain === document.getElementById('domain-select').value);
            
            if (selectedMilestone) {
                requestData.milestone_behavior = selectedMilestone.behavior;
            }
        }
        
        // Send request to the API
        fetch('/comprehensive-assessment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`API responded with status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Display the result
            resultJson.textContent = JSON.stringify(data, null, 2);
            resultContainer.classList.remove('d-none');
            
            // Highlight the result based on score
            const scoreElement = resultContainer.querySelector('.card');
            scoreElement.className = 'card bg-light';
            
            if (data.score !== undefined) {
                if (data.score >= 3) {
                    scoreElement.className = 'card bg-success bg-opacity-10';
                } else if (data.score >= 1) {
                    scoreElement.className = 'card bg-warning bg-opacity-10';
                } else {
                    scoreElement.className = 'card bg-danger bg-opacity-10';
                }
            }
            
            // Scroll to the result
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            // Display error
            resultJson.textContent = JSON.stringify({ error: error.message }, null, 2);
            resultContainer.classList.remove('d-none');
            resultContainer.querySelector('.card').className = 'card bg-danger bg-opacity-10';
        })
        .finally(() => {
            // Reset form state
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Submit Assessment';
        });
    });
    
    // Handle direct test form submission
    document.getElementById('direct-test-form').addEventListener('submit', function(event) {
        event.preventDefault();
        
        const milestoneInput = document.getElementById('direct-milestone');
        const responseInput = document.getElementById('direct-response');
        const submitBtn = document.getElementById('direct-submit-btn');
        const resultContainer = document.getElementById('direct-result-container');
        const resultJson = document.getElementById('direct-result-json');
        
        // Basic validation
        if (!milestoneInput.value.trim()) {
            alert('Please enter a milestone behavior.');
            milestoneInput.focus();
            return;
        }
        
        if (!responseInput.value.trim()) {
            alert('Please enter a parent response.');
            responseInput.focus();
            return;
        }
        
        // Disable form and show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
        
        // Prepare request data
        const requestData = {
            milestone_behavior: milestoneInput.value.trim(),
            parent_response: responseInput.value.trim()
        };
        
        // Send request to the direct test endpoint
        fetch('/comprehensive-assessment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`API responded with status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Display the result
            resultJson.textContent = JSON.stringify(data, null, 2);
            resultContainer.classList.remove('d-none');
            
            // Highlight the result based on score
            const scoreElement = resultContainer.querySelector('.card');
            scoreElement.className = 'card bg-light';
            
            if (data.score !== undefined) {
                if (data.score >= 3) {
                    scoreElement.className = 'card bg-success bg-opacity-10';
                } else if (data.score >= 1) {
                    scoreElement.className = 'card bg-warning bg-opacity-10';
                } else {
                    scoreElement.className = 'card bg-danger bg-opacity-10';
                }
            }
            
            // Scroll to the result
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            // Display error
            resultJson.textContent = JSON.stringify({ error: error.message }, null, 2);
            resultContainer.classList.remove('d-none');
            resultContainer.querySelector('.card').className = 'card bg-danger bg-opacity-10';
        })
        .finally(() => {
            // Reset form state
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Test Direct API';
        });
    });

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