#!/bin/bash

CONTAINER_NAME="mongo-database"

echo "$(tput bold)Initialization script running."
echo -e "$(tput dim)Deleting old Docker containers:$(tput sgr 0)\n"
docker rm -vf $(docker ps -a -q)

echo -e "\n\n$(tput bold dim)Starting Docker containers..$(tput sgr 0)"
docker compose up -d

sleep 10

# Check connectivity
echo -e "\n\n$(tput bold dim)Checking connectivity..$(tput sgr 0)\n"
echo $(docker exec -it $CONTAINER_NAME mongosh --eval "db.runCommand({ ping: 1 })" --quiet | grep 'ok' | awk -v container="$CONTAINER_NAME" -F ': ' '{print container "  ok : " $2}' | tr -d '},')

# Create the table
echo -e "\n\n$(tput bold dim)Setting up database..$(tput sgr 0)"
docker exec -it $CONTAINER_NAME mongosh /app/init-table.js

# Initialize articles
echo -e "\n\n$(tput bold dim)Retrieving article URLs and saving to the database..$(tput sgr 0)"
python init-articles.py