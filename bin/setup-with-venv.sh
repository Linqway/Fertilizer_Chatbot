#!/bin/bash

# Give permission before run:
# chmod +x init.sh
# sudo ./init.sh

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color


present_working_directory=$(pwd)
venv_directory=$(pwd)
nltk_directory=$(pwd)
venv_directory+=/venv/bin/activate
nltk_directory+=/venv/nltk_data

if [ $# -eq 1 ]
  then
    if [[ $1 == --help ]]; then
        printf "\nThese are all available commands in Installation :\n"
        printf "${YELLOW}\t--help\t\t\t${NC}Check flags\n${NC}"
    fi
    exit 0
fi


printf "${YELLOW}\n\nChecking Essential packages...\n\n${NC}"
essential_packages=('python3.8' 'pip' 'virtualenv')
essential_packages_length=${#essential_packages[@]}

packages=0

for i in "${essential_packages[@]}"
do
    printf "${YELLOW}\n\nChecking for ${i}\n\n${NC}"
    files=$(which $i)
    if [[ $? != 0 ]]; then
        printf "\n${RED}$i does not exists!\t${YELLOW}Install $i\n${NC}"
        exit 0
    elif [[ $files ]]; then
        printf "${GREEN}$i exists!\n${NC}"
        packages=$[$packages +1]
    else
        printf "\n${RED}$i does not exists!\t${YELLOW}Install $i\n${NC}"
        exit 0
    fi
done


nltk_installed=false

if [[ $packages -eq $essential_packages_length ]]
then
    printf "${YELLOW}\nCreating virtual environment venv for FertilizerChatbot using python 3.8...${NC}\n\n"
    virtualenv venv --python=python3.8
    source $venv_directory
    printf "${YELLOW}\n\nInstalling dependencies in requirements.txt...${NC}\n\n"
    if [ -e "requirements.txt" ]; then
        pip install -r requirements.txt
        printf "\n${GREEN}Dependencies Installed.${NC}\n\n"
        nltk_installed=true
    else 
        printf "${RED}requirements.txt does not exist in the current directory. Terminating the Installation${NC}\n\n"
        exit 0
    fi 
else
    printf "\n${RED}Essential Packages are not Installed. Terminating the Installation${NC}\n\n"
    exit 0
fi

if $nltk_installed; then
    printf "${YELLOW}\nInstalling required NLTK packages...${NC}\n\n"
    python3 -m nltk.downloader -d ${nltk_directory} wordnet
    python3 -m nltk.downloader -d ${nltk_directory} punkt
    python3 -m nltk.downloader -d ${nltk_directory} vader_lexicon
    python3 -m nltk.downloader -d ${nltk_directory} omw-1.4
    printf "\n\n${GREEN}NLTK packages Installed.${NC}\n\n"
fi

printf "${YELLOW}\nMigrating Database...${NC}\n\n"
python3 src/manage.py migrate
python3 src/manage.py makemigrations
python3 src/manage.py migrate
printf "\n\n${GREEN}Database migrations completed.${NC}\n\n"

printf "\n\n${GREEN}Installation Successfull! Run ${YELLOW}./bin/createuser.sh ${GREEN} to create user!${NC}\n\n"
printf "\n\n${GREEN}Run ${YELLOW}./bin/start-local.sh${GREEN} to start the server!${NC}\n\n"