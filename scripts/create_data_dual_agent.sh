#!/bin/bash

# Exit on error
# set -e

# Fix next seed generation by hash()
export PYTHONHASHSEED=10000

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Print warning with color
print_warning() {
    echo -e "${YELLOW}[Warning] ${1}${NC}"
}

# Create Dual Agent Sokoban dataset
create_dual_agent_sokoban_dataset() {
    print_step "Configuring Dual Agent Sokoban environment settings..."
        
    # Sokoban environment settings for dual agent
    export DIM_X=6
    export DIM_Y=6
    export NUM_BOXES=1

    # generate normal difficulty dataset
    export MAX_STEPS=100
    export SEARCH_DEPTH=30
    
    print_step "Creating Dual Agent Sokoban dataset..."
    print_warning "Note: SOKOBAN errors during creation are normal and can be ignored"
    
    python ragen/env/sokoban/create_dataset.py \
        --output data/sokoban \
        --seed 10000 \
        --train_size 1000 \
        --test_size 100 \
        --prefix qwen-instruct
        
    echo -e "${GREEN}Dual Agent Sokoban dataset created successfully!${NC}"
}

# Main function
main() {
    # Create data directory if it doesn't exist
    mkdir -p data/sokoban
    
    # Create dual agent dataset
    create_dual_agent_sokoban_dataset
    
    echo -e "${GREEN}Dual agent dataset created successfully!${NC}"
}

# Run main function
main 