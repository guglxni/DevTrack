/**
 * Screenshot Generator for ASD Assessment API Demo
 * 
 * This script captures screenshots of the web application for use in presentations.
 * It uses Puppeteer to automate the browser and take screenshots of each tab.
 * 
 * Usage:
 * 1. Make sure the web application is running
 * 2. Run: node take_screenshot.js
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// Create screenshots directory if it doesn't exist
const screenshotsDir = path.join(__dirname, 'screenshots');
if (!fs.existsSync(screenshotsDir)) {
  fs.mkdirSync(screenshotsDir);
}

(async () => {
  console.log('Launching browser...');
  // Launch browser in non-headless mode to ensure proper rendering
  const browser = await puppeteer.launch({
    headless: false,
    defaultViewport: {
      width: 1280,
      height: 800
    }
  });

  const page = await browser.newPage();
  
  // Navigate to the web application
  console.log('Navigating to web application...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  
  // Take a screenshot of the main page
  console.log('Taking screenshot of main page...');
  await page.screenshot({ 
    path: path.join(screenshotsDir, '01_main_page.png'),
    fullPage: true
  });
  
  // Click through each tab and take screenshots
  const tabs = [
    { selector: '#question-tab', filename: '02_question_tab.png' },
    { selector: '#keywords-tab', filename: '03_keywords_tab.png' },
    { selector: '#send-score-tab', filename: '04_send_score_tab.png' },
    { selector: '#score-response-tab', filename: '05_score_response_tab.png' }
  ];
  
  // First tab is already active, so we'll click Send Request to demonstrate a response
  console.log('Demonstrating Question endpoint...');
  await page.click('#sendQuestionBtn');
  // Wait for response to appear
  await page.waitForFunction(
    'document.querySelector("#questionResponse").textContent.includes("success")',
    { timeout: 5000 }
  );
  // Take screenshot with response
  await page.screenshot({ 
    path: path.join(screenshotsDir, '02_question_response.png'),
    fullPage: true
  });
  
  // Click through remaining tabs
  for (let i = 1; i < tabs.length; i++) {
    console.log(`Navigating to ${tabs[i].selector.replace('#', '').replace('-tab', '')} tab...`);
    
    // Click on the tab
    await page.click(tabs[i].selector);
    // Wait for tab content to load
    await page.waitForTimeout(500);
    
    // Take screenshot of the tab
    await page.screenshot({ 
      path: path.join(screenshotsDir, tabs[i].filename),
      fullPage: true
    });
    
    // Click Send Request and take screenshot with response
    const sendBtnSelector = tabs[i].selector.replace('-tab', 'Btn').replace('#score-response', '#sendResponse');
    console.log(`Demonstrating ${tabs[i].selector.replace('#', '').replace('-tab', '')} endpoint...`);
    await page.click(sendBtnSelector);
    
    // Wait for response to appear
    const responseSelector = tabs[i].selector.replace('-tab', 'Response').replace('#score-response', '#response');
    await page.waitForFunction(
      `document.querySelector("${responseSelector}").textContent.includes("success") || document.querySelector("${responseSelector}").textContent.includes("error")`,
      { timeout: 5000 }
    );
    
    // Take screenshot with response
    await page.screenshot({ 
      path: path.join(screenshotsDir, tabs[i].filename.replace('.png', '_response.png')),
      fullPage: true
    });
  }
  
  console.log('All screenshots captured successfully.');
  console.log(`Screenshots saved to: ${screenshotsDir}`);
  
  await browser.close();
})().catch(error => {
  console.error('Error during screenshot capture:', error);
  process.exit(1);
}); 