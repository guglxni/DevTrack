/**
 * ASD Assessment API Demo - Proxy Server
 * 
 * This simple server serves the static files and proxies API requests
 * to overcome potential CORS issues when accessing the API directly
 * from the browser.
 */

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = process.env.API_URL || 'http://localhost:8003';

// Serve static files
app.use(express.static(path.join(__dirname)));

// Proxy API requests
app.use('/', createProxyMiddleware({
    target: API_URL,
    changeOrigin: true,
    pathRewrite: {
        '^/api': ''
    },
    logLevel: 'debug'
}));

// Handle all other routes by serving the index.html
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`ASD Assessment API Demo is running at http://localhost:${PORT}`);
    console.log(`Proxying API requests to ${API_URL}`);
}); 