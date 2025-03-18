#!/bin/bash

# Start the web application
echo "Starting ASD Assessment Web Application..."
echo "Web app will be available at http://localhost:3000"

# Change to the webapp directory
cd webapp

# Check if node_modules exists, if not, install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the web server
npm start

# Exit with the status of the last command
exit $? 