#!/bin/bash
set -e

# This script will run ONLY the very first time the database volume initializes
echo "Initializing secondary MLflow tracking database..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE ${MLFLOW_DB_NAME};
EOSQL
