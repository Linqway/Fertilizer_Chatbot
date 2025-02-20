#!/bin/bash

# Give permission before run:
# chmod +x stop.sh
# sudo ./stop.sh

YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

NOW=$(date +"%T")

printf "Stopping Server...\n"
fuser -k 8000/tcp > /dev/null 2>&1 &
printf "\n${NOW}\t${YELLOW}DJANGO SERVER STOPPED SUCCESSFULLY!\n\n${NC}"