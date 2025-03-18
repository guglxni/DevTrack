#!/bin/bash

# Script to create the monitoring directory for GPU performance logs

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up monitoring directory...${NC}"

# Create the monitoring directory if it doesn't exist
if [ ! -d "monitoring" ]; then
    mkdir -p monitoring
    echo -e "${GREEN}Created monitoring directory.${NC}"
else
    echo -e "${GREEN}Monitoring directory already exists.${NC}"
fi

echo -e "${YELLOW}Monitoring setup complete.${NC}"
echo -e "Performance logs will be stored in the ${GREEN}monitoring/${NC} directory." 