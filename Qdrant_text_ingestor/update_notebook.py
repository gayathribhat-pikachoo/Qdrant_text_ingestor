import json

notebook_path = "Documents Ingestion.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "CSV_FILE" in source and "load_csv" in source and "qdrant.csv" in source:
            cell['source'] = [
                "ORG_ID = \"SELCO_INTERNAL\"  # Enter the target org_id here\n",
                "CSV_FILE = \"data\\\\qdrant.csv\"\n",
                "all_records = load_csv(CSV_FILE)\n",
                "records = [r for r in all_records if r.get('org_id') == ORG_ID]\n",
                "print(f\"Loaded {len(records)} records for org_id: {ORG_ID}\")"
            ]
            # remove outputs to make it clean if we want, or keep it.
            # cell['outputs'] = [] 
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
    
print("Notebook updated.")
