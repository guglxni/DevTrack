// Function to fetch all milestones from the API
function fetchMilestones() {
    console.log('Fetching milestones from /milestones');
    fetch('/milestones')
        .then(response => {
            console.log('Response status:', response.status);
            return response.json();
        })
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