#!/bin/bash

# Setup script for Metal GPU acceleration and monitoring tools

# Set text colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Metal GPU Acceleration for LLM Inference${NC}"
echo "======================================================"

# Function to detect Apple Silicon
detect_apple_silicon() {
    if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        echo -e "${GREEN}✅ Apple Silicon detected${NC}"
        
        # Determine specific chip model
        local chip_info=$(sysctl -n machdep.cpu.brand_string)
        if [[ $chip_info == *"M1"* ]]; then
            echo -e "${BLUE}→ Apple M1 chip detected${NC}"
        elif [[ $chip_info == *"M2"* ]]; then
            echo -e "${BLUE}→ Apple M2 chip detected${NC}"
        elif [[ $chip_info == *"M3"* ]]; then
            echo -e "${BLUE}→ Apple M3 chip detected${NC}"
        elif [[ $chip_info == *"M4"* ]]; then
            echo -e "${BLUE}→ Apple M4 chip detected${NC}"
        else
            echo -e "${YELLOW}⚠️ Unknown Apple Silicon variant detected${NC}"
        fi
        return 0
    else
        echo -e "${YELLOW}⚠️ Not running on Apple Silicon${NC}"
        echo -e "${YELLOW}Metal GPU acceleration is only available on Apple Silicon Macs.${NC}"
        echo -e "${YELLOW}Setup will continue, but acceleration will not be available.${NC}"
        return 1
    fi
}

# Make all scripts executable
make_scripts_executable() {
    echo -e "\n${BLUE}Making scripts executable...${NC}"
    chmod +x start_optimized.sh
    chmod +x restart_with_metal.sh
    chmod +x restart_with_metal_advanced.sh
    chmod +x start_server.sh
    chmod +x analyze_performance.py
    chmod +x benchmark_llm.py

    echo -e "${GREEN}✅ All scripts are now executable${NC}"
}

# Install Python dependencies
install_dependencies() {
    echo -e "\n${BLUE}Installing Python dependencies for monitoring tools...${NC}"
    
    if command -v pip3 &> /dev/null; then
        pip3 install -r gpu_monitoring_requirements.txt
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
        else
            echo -e "${RED}❌ Failed to install dependencies${NC}"
            echo -e "${YELLOW}Please install them manually: pip3 install -r gpu_monitoring_requirements.txt${NC}"
        fi
    else
        echo -e "${RED}❌ pip3 not found${NC}"
        echo -e "${YELLOW}Please install pip3 and then run: pip3 install -r gpu_monitoring_requirements.txt${NC}"
    fi
}

# Create monitoring directory
create_directories() {
    echo -e "\n${BLUE}Creating required directories...${NC}"
    mkdir -p monitoring
    echo -e "${GREEN}✅ Created monitoring directory${NC}"
}

# Function to modify existing shell profiles to include Metal env vars
update_shell_profile() {
    echo -e "\n${BLUE}Would you like to add Metal GPU acceleration variables to your shell profile?${NC}"
    echo -e "${YELLOW}This will make Metal acceleration available by default.${NC}"
    read -p "Update shell profile? (y/n) [n]: " update_profile
    
    if [[ "$update_profile" == "y" || "$update_profile" == "Y" ]]; then
        # Determine which shell profile to update
        if [[ -f "$HOME/.zshrc" ]]; then
            profile="$HOME/.zshrc"
        elif [[ -f "$HOME/.bash_profile" ]]; then
            profile="$HOME/.bash_profile"
        elif [[ -f "$HOME/.bashrc" ]]; then
            profile="$HOME/.bashrc"
        else
            echo -e "${YELLOW}Could not find a shell profile to update.${NC}"
            return
        fi
        
        # Add variables to profile
        echo -e "\n# Metal GPU Acceleration for LLM" >> "$profile"
        echo "export USE_METAL=true" >> "$profile"
        echo "export N_GPU_LAYERS=32" >> "$profile"
        echo "export GGML_METAL_FULL_OFFLOAD=1" >> "$profile"
        echo "export F16_KV=true" >> "$profile"
        
        echo -e "${GREEN}✅ Updated $profile with Metal acceleration variables${NC}"
        echo -e "${YELLOW}Please restart your terminal or run 'source $profile' to apply changes${NC}"
    else
        echo -e "${BLUE}Skipping shell profile update${NC}"
    fi
}

# Main execution flow
detect_apple_silicon
make_scripts_executable
install_dependencies
create_directories

# Only offer to update shell profile if running on Apple Silicon
if [[ $? -eq 0 ]]; then
    update_shell_profile
fi

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "${BLUE}To start the server with Metal GPU acceleration, run:${NC}"
echo -e "  ${YELLOW}./start_optimized.sh${NC}"
echo -e "\n${BLUE}For more information, read:${NC}"
echo -e "  ${YELLOW}GPU_ACCELERATION.md${NC}" 