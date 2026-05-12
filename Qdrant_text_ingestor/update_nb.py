import json

notebook_path = "Documents Ingestion.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update the embedding generation cell to include org_id filter
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('final_data=[]' in line for line in cell['source']):
        new_source = [
            "TARGET_ORG_ID = None  # Enter an org_id here to only process records for that organization (e.g., 'SELCO_INTERNAL')\n",
            "\n",
            "final_data=[]\n",
            "for n,chunk in enumerate(records):\n",
            "    if TARGET_ORG_ID and chunk.get('org_id') != TARGET_ORG_ID:\n",
            "        continue\n",
            "        \n",
            "    print(f\"Processing {n+1} record\")\n",
            "    \n",
            "    # Embedding the chunks\n",
            "    chunk['embedding']=  get_openai_embedding(chunk[\"text\"])\n",
            "    final_data.append(chunk)"
        ]
        cell['source'] = new_source
        break

# 2. Add a new cell to save embeddings to CSV
# Find the index of the cell that has "final_data" (the one outputting the data)
insert_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell['source'] == ['final_data']:
        insert_idx = i + 1
        break

new_markdown_cell = {
    "cell_type": "markdown",
    "id": "save-csv-markdown",
    "metadata": {},
    "source": [
        "# Save embeddings to CSV"
    ]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "save-csv-code",
    "metadata": {},
    "outputs": [],
    "source": [
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
}

nb['cells'].insert(insert_idx, new_code_cell)
nb['cells'].insert(insert_idx, new_markdown_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
