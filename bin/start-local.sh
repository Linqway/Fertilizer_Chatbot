#!/bin/bash

# Give permission before run:
# chmod +x start.sh
# sudo ./start.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

NOW=$(date +"%T")

printf "Starting Server...\n"
python src/manage.py crontab add > /dev/null 2>&1 &
python src/manage.py runserver > /dev/null 2>&1 &
printf "\n${NOW}\t${GREEN}DJANGO SERVER STARTED SUCCESSFULLY!\n${NC}"
printf "\n${YELLOW}Application running in http://localhost:8000/\n\n${NC}"