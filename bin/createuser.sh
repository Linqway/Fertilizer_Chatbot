#!/bin/bash

# Give permission before run:
# chmod +x init.sh
# sudo ./init.sh

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color


printf "${YELLOW}\n\nCreating Users for the App!...\n\n${NC}"
python3 FertilizerChatbot/configurations/createUser.py
printf "\n\n${GREEN}Created Super User!${NC}\n\n"
