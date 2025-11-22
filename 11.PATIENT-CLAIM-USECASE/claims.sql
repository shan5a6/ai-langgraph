CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE claims (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id INT NOT NULL,
    status VARCHAR(50) NOT NULL,
    decision_details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

select * from claims

docker exec -it my_postgres psql -U myuser -d postgres

postgres=# CREATE DATABASE claims_db;
CREATE DATABASE


