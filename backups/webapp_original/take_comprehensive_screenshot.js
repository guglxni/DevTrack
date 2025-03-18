/**
 * Screenshot Generator for ASD Assessment API Demo
 * 
 * This script uses Puppeteer to take screenshots of the web application,
 * capturing different API endpoints and responses.
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// Create the img directory if it doesn't exist
const imgDir = path.join(__dirname, 'img');
if (!fs.existsSync(imgDir)) {
  fs.mkdirSync(imgDir);
}

(async () => {
  console.log('Launching browser...');
  const browser = await puppeteer.launch({
    headless: true,
    defaultViewport: {
      width: 1280,
      height: 800
    }
  });

  try {
    console.log('Opening new page...');
    const page = await browser.newPage();

    // Visit the application
    console.log('Navigating to the web application...');
    await page.goto('http://localhost:3000', {
      waitUntil: 'networkidle2',
      timeout: 30000 // 30 seconds timeout
    });

    // Take screenshot of the landing page
    console.log('Taking screenshot of the landing page...');
    await page.screenshot({ path: path.join(imgDir, 'screenshot_landing.png') });

    // Take screenshots of each tab
    const tabs = [
      { id: '#question-tab', name: 'question' },
      { id: '#keywords-tab', name: 'keywords' },
      { id: '#send-score-tab', name: 'send_score' },
      { id: '#score-response-tab', name: 'score_response' },
      { id: '#comprehensive-tab', name: 'comprehensive' }
    ];

    for (const tab of tabs) {
      console.log(`Clicking on ${tab.name} tab...`);
      await page.click(tab.id);
      
      // Wait for the tab to be fully loaded
      await page.waitForTimeout(500);
      
      console.log(`Taking screenshot of ${tab.name} tab...`);
      await page.screenshot({ path: path.join(imgDir, `screenshot_${tab.name}_tab.png`) });

      // If it's the comprehensive tab, also take a screenshot with the form filled out
      if (tab.name === 'comprehensive') {
        // Fill in form fields
        console.log('Filling in comprehensive assessment form...');
        await page.type('#compQuestion', 'Does the child recognize familiar people?');
        await page.type('#compMilestoneBehavior', 'Recognizes familiar people');
        
        // Clear and fill the parent response field
        await page.click('#compParentResponse', {clickCount: 3}); // Select all
        await page.keyboard.press('Backspace'); // Delete
        await page.type('#compParentResponse', 'My child always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well.');
        
        // Ensure keywords are shown
        const isChecked = await page.$eval('#includeKeywords', el => el.checked);
        if (!isChecked) {
          await page.click('#includeKeywords');
          await page.waitForTimeout(300);
        }
        
        // Take screenshot of the form filled out
        console.log('Taking screenshot of filled comprehensive form...');
        await page.screenshot({ path: path.join(imgDir, 'screenshot_comprehensive_filled.png') });
        
        // Submit the form and wait for response
        console.log('Submitting comprehensive form...');
        await page.click('#sendComprehensiveBtn');
        
        // Wait for the response to be displayed
        await page.waitForFunction(
          () => document.querySelector('#comprehensiveResponse').innerText.includes('score_label'),
          { timeout: 10000 }
        );
        
        // Take screenshot showing the response
        console.log('Taking screenshot of comprehensive response...');
        await page.screenshot({ path: path.join(imgDir, 'screenshot_comprehensive_response.png') });
      }
    }

    console.log('All screenshots have been saved to the img directory!');
  } catch (error) {
    console.error('Error taking screenshots:', error);
  } finally {
    await browser.close();
  }
})(); 