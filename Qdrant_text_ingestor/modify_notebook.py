import json

notebook_path = "Documents Ingestion.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

new_source = [
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "if final_data:\n",
    "    df_new = pd.DataFrame(final_data)\n",
    "    file_path = 'data/qdrant.csv'\n",
    "    \n",
    "    if os.path.exists(file_path):\n",
    "        df_existing = pd.read_csv(file_path)\n",
    "        if 'embedding' not in df_existing.columns:\n",
    "            df_existing['embedding'] = None\n",
    "        \n",
    "        df_existing.set_index('uid', inplace=True)\n",
    "        df_new.set_index('uid', inplace=True)\n",
    "        \n",
    "        df_existing.update(df_new[['embedding']])\n",
    "        df_existing.reset_index(inplace=True)\n",
    "        \n",
    "        df_existing.to_csv(file_path, index=False)\n",
    "        print(f\"Updated {len(df_new)} records with embeddings in {file_path}\")\n",
    "    else:\n",
    "        df_new.to_csv(file_path, index=False)\n",
    "        print(f\"Created {file_path} with {len(df_new)} records\")\n",
    "else:\n",
    "    print(\"No data to save.\")\n"
]

for cell in notebook['cells']:
    if cell.get('id') == 'save-csv-code' or (cell['cell_type'] == 'code' and "embeddings_output.csv" in "".join(cell.get('source', []))):
        cell['source'] = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
    
print("Notebook updated to filter existing uids.")
