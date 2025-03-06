/**
 * ASD Developmental Milestone Assessment API - JavaScript Example
 * 
 * This example demonstrates how to interact with the API from JavaScript.
 * Useful for integrating with web applications, SPAs, or Node.js services.
 */

// API configuration
const API_BASE_URL = 'http://localhost:8000';

/**
 * Sets the child's age in months
 * @param {number} age - Child's age in months (0-36)
 * @returns {Promise<Object>} - API response
 */
async function setChildAge(age) {
  try {
    const response = await fetch(`${API_BASE_URL}/set-child-age`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ age })
    });
    return await response.json();
  } catch (error) {
    console.error('Error setting child age:', error);
    throw error;
  }
}

/**
 * Gets the next milestone to assess
 * @returns {Promise<Object>} - API response with milestone data
 */
async function getNextMilestone() {
  try {
    const response = await fetch(`${API_BASE_URL}/next-milestone`);
    return await response.json();
  } catch (error) {
    console.error('Error getting next milestone:', error);
    throw error;
  }
}

/**
 * Scores a response for a specific milestone
 * @param {string} milestone_behavior - The behavior being assessed
 * @param {string} response - Caregiver's response describing the child's behavior
 * @returns {Promise<Object>} - API response with score data
 */
async function scoreResponse(milestone_behavior, response) {
  try {
    const response_data = await fetch(`${API_BASE_URL}/score-response`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        milestone_behavior, 
        response 
      })
    });
    return await response_data.json();
  } catch (error) {
    console.error('Error scoring response:', error);
    throw error;
  }
}

/**
 * Scores multiple responses in batch for better performance
 * @param {Array<Object>} responses - Array of response objects with milestone_behavior and response
 * @returns {Promise<Array<Object>>} - API response with score data
 */
async function batchScoreResponses(responses) {
  try {
    const response = await fetch(`${API_BASE_URL}/batch-score`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ responses })
    });
    return await response.json();
  } catch (error) {
    console.error('Error batch scoring responses:', error);
    throw error;
  }
}

/**
 * Generates a comprehensive assessment report
 * @returns {Promise<Object>} - API response with report data
 */
async function generateReport() {
  try {
    const response = await fetch(`${API_BASE_URL}/generate-report`);
    return await response.json();
  } catch (error) {
    console.error('Error generating report:', error);
    throw error;
  }
}

/**
 * Resets the assessment engine for a new assessment
 * @returns {Promise<Object>} - API response
 */
async function resetAssessment() {
  try {
    const response = await fetch(`${API_BASE_URL}/reset`, {
      method: 'POST'
    });
    return await response.json();
  } catch (error) {
    console.error('Error resetting assessment:', error);
    throw error;
  }
}

/**
 * Example full assessment flow
 */
async function runFullAssessment() {
  try {
    // Reset any previous assessment
    await resetAssessment();
    
    // Set child's age
    const ageResult = await setChildAge(24);
    console.log(`Assessment started with ${ageResult.total_milestones} milestones`);
    
    // Sample responses for batch processing
    const sampleResponses = [
      {
        milestone_behavior: "Opens mouth for spoon",
        response: "My child always opens their mouth when they see the spoon coming."
      },
      {
        milestone_behavior: "Plays with toys",
        response: "My child enjoys playing with a variety of toys, especially blocks."
      },
      {
        milestone_behavior: "Follows simple directions",
        response: "When I ask my child to come here, they usually understand and follow."
      }
    ];
    
    // Score responses in batch
    const scoreResults = await batchScoreResponses(sampleResponses);
    console.log('Batch scoring results:', scoreResults);
    
    // Generate final report
    const report = await generateReport();
    console.log('Domain quotients:', report.domain_quotients);
    console.log(`Assessment complete with ${report.scores.length} scored milestones`);
    
  } catch (error) {
    console.error('Assessment error:', error);
  }
}

// Node.js example usage
if (typeof require !== 'undefined') {
  // Only run if in Node.js environment
  runFullAssessment().then(() => {
    console.log('Assessment demo complete');
  });
}

// Browser export
if (typeof window !== 'undefined') {
  // Make functions available globally in browser
  window.asdApi = {
    setChildAge,
    getNextMilestone,
    scoreResponse,
    batchScoreResponses,
    generateReport,
    resetAssessment,
    runFullAssessment
  };
} 