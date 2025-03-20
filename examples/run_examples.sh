#!/bin/bash

# StockScan Examples Runner
# This script helps run the StockScan example scripts

set -e  # Exit on error

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}     StockScan Examples Runner    ${NC}"
echo -e "${BLUE}==================================${NC}"
echo ""

# Check if the StockScan package is installed
if python -c "import stockscan" &> /dev/null; then
    echo -e "${GREEN}✓ StockScan package is installed${NC}"
else
    echo -e "${RED}✗ StockScan package is not installed${NC}"
    echo -e "${YELLOW}Installing StockScan package...${NC}"
    pip install -e .
    echo -e "${GREEN}✓ StockScan package installed${NC}"
fi

# Check if .env file exists
if [ -f .env ]; then
    echo -e "${GREEN}✓ .env file found${NC}"
else
    echo -e "${YELLOW}Creating .env file...${NC}"
    echo "# Add your OpenAI API key here" > .env
    echo "OPENAI_API_KEY=" >> .env
    echo "CHAT_MODEL=gpt-4-turbo" >> .env
    echo -e "${RED}⚠ Please add your OpenAI API key to the .env file${NC}"
fi

# Check for Qlib data
US_DATA_PATH=~/.qlib/qlib_data/us_data
if [ -d "$US_DATA_PATH" ]; then
    echo -e "${GREEN}✓ US stock data found${NC}"
else
    echo -e "${RED}✗ US stock data not found${NC}"
    echo -e "${YELLOW}Do you want to download US stock data? (y/n)${NC}"
    read -r download_data
    if [[ $download_data == "y" ]]; then
        echo -e "${YELLOW}Downloading US stock data (this may take a while)...${NC}"
        python -m qlib.data.data_collector_us
        echo -e "${GREEN}✓ US stock data downloaded${NC}"
    else
        echo -e "${RED}⚠ US stock data is required for the examples${NC}"
    fi
fi

# Check for Docker (required for RDAgent)
if command -v docker &> /dev/null && docker info &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed and running${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ Docker is not available. RDAgent evolution features will be disabled.${NC}"
    DOCKER_AVAILABLE=false
fi

echo ""
echo -e "${BLUE}Available Examples:${NC}"
echo -e "1. ${GREEN}Trio Stocks Prediction${NC} - AAPL, MSFT, NVDA predictions"
echo -e "2. ${GREEN}Tech Stocks Prediction${NC} - Comprehensive tech stock analysis"
echo -e "3. ${GREEN}Simple Pipeline${NC} - Basic StockScan pipeline example"
echo -e "q. ${RED}Quit${NC}"
echo ""

echo -e "${YELLOW}Enter the number of the example you want to run:${NC}"
read -r choice

case $choice in
    1)
        echo -e "${BLUE}Running Trio Stocks Prediction...${NC}"
        python examples/trio_stocks_prediction.py
        ;;
    2)
        echo -e "${BLUE}Running Tech Stocks Prediction...${NC}"
        python examples/tech_stocks_prediction.py
        ;;
    3)
        echo -e "${BLUE}Running Simple Pipeline...${NC}"
        python examples/simple_pipeline.py
        ;;
    q)
        echo -e "${BLUE}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        ;;
esac

echo ""
echo -e "${GREEN}Example completed!${NC}"
echo -e "${BLUE}==================================${NC}" 