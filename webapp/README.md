# ASD Assessment API Web Demo Application

## Overview

This web application provides a user-friendly interface for demonstrating the capabilities of the ASD Assessment API. It showcases all four API endpoints through an intuitive, responsive interface designed for presentations and stakeholder demonstrations.

## Features

- **Tabbed Interface**: Organized into four tabs, one for each API endpoint
- **Interactive Forms**: User-friendly input forms for each endpoint
- **Visual Feedback**: Response time indicators and formatted JSON responses
- **Error Handling**: Clear error messages and recovery options
- **Responsive Design**: Works on desktop and mobile devices
- **Proxy Integration**: Built-in proxy to handle CORS issues with the API

## Architecture

The web demo application consists of three main components:

1. **Frontend Interface**: HTML, CSS, and JavaScript providing the user interface
2. **Node.js Server**: Express.js server hosting the static assets
3. **Proxy Middleware**: HTTP-Proxy-Middleware handling API requests

## Prerequisites

- Node.js 14.x or higher
- npm 6.x or higher
- ASD Assessment API running on port 8003

## Installation

1. Install dependencies:

```bash
cd webapp
npm install
```

2. Ensure the API server is running:

```bash
cd src/api
uvicorn main:app --reload --port 8003
```

3. Start the web application:

```bash
cd webapp
npm start
```

4. Access the application at `http://localhost:3000`

## API Endpoint Demonstrations

### 1. Question Endpoint

The Question tab demonstrates the API's ability to process and match questions related to developmental milestones.

**Features**:
- Input field for question text
- Optional milestone ID field
- Displays matching results with confidence score
- Shows API response time

**Example Usage**:
1. Enter a question like "Does the child respond when called by name?"
2. Optionally provide a milestone ID like "Recognizes familiar people"
3. Click "Send Request"
4. View the structured response with matching information

### 2. Keywords Endpoint

The Keywords tab showcases the API's ability to manage categorized keywords used for response analysis.

**Features**:
- Dropdown for keyword categories
- Text area for entering keywords (comma-separated)
- Displays success message with keyword count
- Shows API response time

**Example Usage**:
1. Select a category (e.g., "CANNOT_DO")
2. Enter keywords like "no, not, never, doesn't, cannot, can't"
3. Click "Send Request"
4. View the confirmation response with category and count information

### 3. Send Score Endpoint

The Send Score tab demonstrates the API's ability to record scores for specific developmental milestones.

**Features**:
- Input field for milestone ID
- Numeric input for score (0-4)
- Automatic score label based on selected score
- Shows API response time

**Example Usage**:
1. Enter a milestone ID like "Shows empathy"
2. Select a score (e.g., 4)
3. Note that the score label updates automatically to "INDEPENDENT"
4. Click "Send Request"
5. View the confirmation response

### 4. Score Response Endpoint

The Score Response tab demonstrates the API's most powerful feature - analyzing parent/caregiver responses to determine appropriate scores.

**Features**:
- Input field for milestone behavior
- Text area for parent/caregiver response
- Displays determined score with confidence level
- Shows API response time

**Example Usage**:
1. Enter a milestone behavior like "Recognizes familiar people"
2. Enter a detailed response like "My child always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well."
3. Click "Send Request"
4. View the analyzed response with score, score label, and confidence level

## API Response Processing

The application processes API responses in the following way:

1. Requests are sent through the proxy middleware to avoid CORS issues
2. Response times are calculated and displayed
3. JSON responses are formatted for readability
4. Success/error states are visually indicated
5. Error messages are displayed in a user-friendly format

## Files and Directories

- `index.html`: Main HTML structure with tabs for each endpoint
- `css/styles.css`: CSS styles for the application
- `js/app.js`: JavaScript for handling API interactions and UI updates
- `server.js`: Express server with proxy middleware
- `package.json`: Project dependencies and scripts
- `img/`: Directory containing application screenshots and images
- `demo_presentation.md`: Guide for presenting the application

## NLP Processing and Fallback Mechanism

The Score Response endpoint demonstration showcases two analysis methods:

1. **Advanced NLP Analysis**: When available, sophisticated natural language processing analyzes the semantic content of responses.

2. **Fallback Pattern Detection**: If the advanced NLP module is unavailable, the system falls back to pattern detection using keywords, ensuring continuous operation.

The demo transparently shows when the fallback mechanism is being used, demonstrating the system's resilience.

## Running with the API

For the best demonstration experience, both the API server and web application should be running simultaneously. You can use the convenience script:

```bash
./scripts/start_demo.sh
```

This starts both the API server and web application in the background.

## Troubleshooting

### Common Issues:

1. **Cannot Access Web Application**
   - Ensure Node.js server is running
   - Try using `http://127.0.0.1:3000` instead of `localhost`
   - Check for console errors in browser developer tools

2. **API Connection Errors**
   - Verify API server is running on port 8003
   - Check terminal for API server errors
   - Verify proxy configuration in `server.js`

3. **Port Already in Use**
   - Web app port (3000): Check with `lsof -i :3000`
   - API port (8003): Check with `lsof -i :8003`
   - Kill existing processes if needed: `kill -9 [PID]`

4. **Node.js Errors**
   - Ensure correct Node.js version (14+)
   - Try reinstalling dependencies: `npm ci`
   - Check for syntax errors in JavaScript files

## Customization

You can customize the application by:

1. **Modifying Sample Data**: Edit the placeholder values in `index.html`
2. **Changing Styles**: Modify `css/styles.css` for visual changes
3. **Adjusting Proxy Configuration**: Edit `server.js` for different API endpoints or ports

## Script Documentation

The application includes several scripts in `package.json`:

- `npm start`: Starts the Express server
- `npm run dev`: Starts the server with nodemon for auto-reloading
- `npm run screenshot`: Runs the screenshot capture script for presentations

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. 