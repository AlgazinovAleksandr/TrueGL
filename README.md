# TrueGL Database

The system is deployed using __Docker__ for containerization and __MongoDB__ for structured data (1 relational table – `Articles`).

## Setup

VPN must be on. <br />
Download the project <br />
`git clone --branch database git@github.com:AlgazinovAleksandr/TrueGL.git` <br />
Install required Python libraries <br />
`pip install requests beautifulsoup4 pymongo`

### Windows

Install [__Docker Desktop__](https://www.docker.com/products/docker-desktop/) <br />
TODO

### Linux

Install Docker + Docker Compose <br />
`sudo apt-get install docker docker-compose` <br />
Download MongoDB image <br />
`docker pull mongo:latest` <br />
Run the initialization script <br />
`cd TrueGL` <br />
`chmod +x init.sh` (to make init.sh script executable) <br />
`./init.sh` (automates data center setup)

To start the database for subsequent launches <br />
`docker compose up` <br />

The database can also be connected to and managed using [__MongoDB Compass__](https://www.mongodb.com/products/tools/compass). To connect, create a new connection and use the following URI: `mongodb://localhost:27017/`.
