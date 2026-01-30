
import json
import sys

NODE_ID = "a0ed5aa5-e9b1-446f-973e-8004df17cbc3"

try:
    with open('/opt/n8n-clipper/tmp/workflow.json', 'r') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    for node in nodes:
        if node['id'] == NODE_ID:
            print(json.dumps(node.get('parameters', {}), indent=2))
            break
            
except Exception as e:
    print(f"Error: {e}")
