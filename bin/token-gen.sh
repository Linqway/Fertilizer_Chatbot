#!/bin/bash

# Give permission before run:
# chmod +x init.sh
# sudo ./init.sh

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

present_working_directory=$(pwd)
present_working_directory+=/venv/bin/activate


printf "${YELLOW}\n\nCreating SECRET TOKEN for the App!...\n\n${NC}"
source $present_working_directory
python3 FertilizerChatbot/configurations/token-gen.py
printf "\n\n${GREEN}DONE!${NC}\n\n"
