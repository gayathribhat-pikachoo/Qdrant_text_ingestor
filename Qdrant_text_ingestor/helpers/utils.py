import os
import pickle
import re
from typing import Any

import pandas as pd
import json
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


MARKDOWN_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n##### ",
    "\n###### ",
    "\n\n",
    "\n",
    " ",
]


def _text_from_doc(value):
    if not isinstance(value, str) or not value.strip():
        return None

    for candidate in (value, value.replace('""', '"')):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return text

    return None


def load_csv(file_path):
    """
    Load a CSV file and return a list of dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
    
    df = pd.read_csv(file_path)
    if "text" in df.columns and "doc" in df.columns:
        missing_text = df["text"].isna() | df["text"].astype(str).str.strip().eq("")
        if missing_text.any():
            df.loc[missing_text, "text"] = df.loc[missing_text, "doc"].apply(_text_from_doc)
    return df.to_dict('records')

def load_pickle(file_path):
    """
    Load a pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
        
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def batch_list(input_list, batch_size):
    """
    Split a list into smaller batches.
    """
    return [input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)]


def count_tokens_str(doc: str) -> int:
    """Count text tokens using the same practical token sizing used for chunk windows."""
    if not doc or not str(doc).strip():
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(doc)))


def clean_markdown_text(text: Any) -> str:
    """
    Clean markdown text by removing images and excess whitespace before chunking.
    """
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"!\[.*?\]\(data:image.*?\)", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"<img.*?>", "", text)
    return text.strip()


def get_sliding_windows_for_markdown(
    text: Any,
    window_size: int = 400,
    overlap: int = 80,
    min_tokens: int = 100,
) -> list[str]:
    """
    Split markdown into overlapping windows, preferring markdown header boundaries.
    """
    text = clean_markdown_text(text)
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=MARKDOWN_SEPARATORS,
        keep_separator=True,
        chunk_size=window_size,
        chunk_overlap=overlap,
        length_function=count_tokens_str,
    )

    return [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if chunk.strip() and count_tokens_str(chunk) >= min_tokens
    ]


def chunk_record_for_embedding(
    record: dict[str, Any],
    text_field: str = "text",
    window_size: int = 400,
    overlap: int = 80,
    min_tokens: int = 100,
) -> list[dict[str, Any]]:
    """Return one record per markdown-aware sliding-window chunk."""
    chunks = get_sliding_windows_for_markdown(
        record.get(text_field),
        window_size=window_size,
        overlap=overlap,
        min_tokens=min_tokens,
    )
    if not chunks:
        return []

    base_uid = str(record.get("uid", "")).strip()
    chunked_records = []
    for index, chunk in enumerate(chunks, start=1):
        item = dict(record)
        item[text_field] = chunk
        item["chunk_index"] = index
        item["chunk_count"] = len(chunks)

        if len(chunks) > 1 and base_uid:
            item["parent_uid"] = base_uid
            item["uid"] = f"{base_uid}_{index}"

        chunked_records.append(item)

    return chunked_records


def chunk_records_for_embedding(
    records: list[dict[str, Any]],
    text_field: str = "text",
    window_size: int = 400,
    overlap: int = 80,
    min_tokens: int = 100,
) -> list[dict[str, Any]]:
    """Chunk a list of records using markdown-aware sliding windows."""
    chunked_records = []
    for record in records:
        chunked_records.extend(
            chunk_record_for_embedding(
                record,
                text_field=text_field,
                window_size=window_size,
                overlap=overlap,
                min_tokens=min_tokens,
            )
        )
    return chunked_records
