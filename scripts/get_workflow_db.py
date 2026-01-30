
import psycopg2
import os
import json

# Database config from .env
DB_HOST = "localhost" # accessed via localhost since we run this on the host or inside the n8n container? 
# Wait, if I run this on the host (VPS), I need to access the postgres container.
# The docker-compose shows postgres-prod is on the 'production' network.
# I can run this script INSIDE the n8n container where it has access to postgres-prod.

# Configuration for inside the container/network
DB_DSN = "host=postgres-prod port=5432 dbname=n8n_db user=n8n_user password=be323a93d39126f1574b149cbaa04ac8"

try:
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    # Query to get the workflow data
    cur.execute("SELECT nodes FROM workflow_entity WHERE id = '6k9pvd1m902quB7Zev367';")
    row = cur.fetchone()
    
    if row:
        nodes = row[0]
        # It comes as a string or json object depending on the driver, usually string in n8n db context or jsonb
        print(json.dumps(nodes, indent=2))
    else:
        print("Workflow not found")
        
    cur.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")
