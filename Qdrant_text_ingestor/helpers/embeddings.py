import math
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv
from helpers.utils import batch_list
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _as_embedding_api_string(value: Any) -> str:
    """Coerce embedding input to a plain string; pandas NaN becomes empty (invalid for API JSON)."""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (ImportError, TypeError, ValueError):
        pass
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, str):
        s = value.strip()
        return "" if not s or s.lower() == "nan" else s
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "<na>", "nat"):
        return ""
    return s


def get_openai_embedding(text, model="text-embedding-3-large", **kwargs):
    """
    Get the embeddings of the text from OpenAI API.

    Args:
        text (str): Text to get embeddings for.
        model (str): Model to use for embeddings.
    Returns:
        list: Embeddings of the text.
    """
    s = _as_embedding_api_string(text)
    if not s:
        raise ValueError(
            "Cannot embed empty or missing text (NaN/None/blank). "
            "Drop or fix the row before calling the embeddings API."
        )
    response = openai_client.embeddings.create(
        model=model,
        input=[s],
        **kwargs,
    )
    return response.data[0].embedding


def get_batch_openai_embedding(texts: list, model="text-embedding-3-large", **kwargs):
    """
    Get embeddings of a batch of texts from OpenAI API.

    Args:
        texts (list): List of texts to get embeddings for.
        model (str): Model to use for embeddings.
        **kwargs: Additional arguments to pass to the OpenAI API.
    Returns:
        list[list]: List of embeddings of the texts.
    """
    cleaned = [_as_embedding_api_string(t) for t in texts]
    bad = [i for i, s in enumerate(cleaned) if not s]
    if bad:
        raise ValueError(
            f"Cannot embed {len(bad)} empty/NaN text(s) at indices {bad[:10]}{'...' if len(bad) > 10 else ''}. "
            "Filter those rows before batch embedding."
        )
    text_batches = batch_list(cleaned, 1024) if len(cleaned) > 1024 else [cleaned]
    embeddings = []
    for text_batch in text_batches:
        response = openai_client.embeddings.create(
            model=model,
            input=text_batch,
            **kwargs,
        )
        embeddings += [r.embedding for r in response.data]
    return embeddings