
import json
import sys

try:
    with open('/opt/n8n-clipper/tmp/workflow.json', 'r') as f:
        data = json.load(f)
    
    nodes = data
    if isinstance(data, dict) and 'nodes' in data:
        nodes = data['nodes']
        
    for node in nodes:
        if node['name'] == 'On form submission':
            print(f"Trigger {node['id']}")
        elif node['name'] == 'Basic LLM Chain':
            print(f"LLM {node['id']}")
            
except Exception as e:
    print(f"Error parsing JSON: {e}")
