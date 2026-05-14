import math
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from .utils import batch_list
load_dotenv()

openai_client: Any | None = None
FASTEMBED_MODEL_NAME = "intfloat/multilingual-e5-large"


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


def get_openai_client() -> Any:
    """Create the OpenAI client lazily so FastEmbed-only runs do not need an API key."""
    from openai import OpenAI

    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return openai_client


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
    response = get_openai_client().embeddings.create(
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
    if isinstance(texts, (str, bytes)) or not hasattr(texts, "__iter__"):
        raise ValueError(
            "get_batch_openai_embedding expects a list of text values. "
            "Use get_openai_embedding(text) for one record, or pass [text1, text2, ...]."
        )

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
        response = get_openai_client().embeddings.create(
            model=model,
            input=text_batch,
            **kwargs,
        )
        embeddings += [r.embedding for r in response.data]
    return embeddings


@lru_cache(maxsize=2)
def get_fastembed_model(model_name: str = FASTEMBED_MODEL_NAME):
    """Load and cache the FastEmbed model so notebook reruns do not reload it per batch."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(
            "fastembed is required for local embedding generation. "
            "Install it with `uv add fastembed` in Qdrant_text_ingestor."
        ) from exc

    return TextEmbedding(model_name=model_name)


def _with_e5_prefix(text: str, input_type: str) -> str:
    prefix = input_type.strip().lower()
    if prefix not in {"passage", "query", ""}:
        raise ValueError("input_type must be 'passage', 'query', or ''.")
    if not prefix:
        return text
    if text.lower().startswith(("passage: ", "query: ")):
        return text
    return f"{prefix}: {text}"


def get_fastembed_embedding(
    text,
    model: str = FASTEMBED_MODEL_NAME,
    input_type: str = "passage",
) -> list[float]:
    """
    Get one embedding using FastEmbed.

    E5 models expect ``passage:`` for documents and ``query:`` for queries.
    """
    return get_batch_fastembed_embedding([text], model=model, input_type=input_type)[0]


def get_batch_fastembed_embedding(
    texts: list,
    model: str = FASTEMBED_MODEL_NAME,
    batch_size: int = 100,
    input_type: str = "passage",
) -> list[list[float]]:
    """
    Get embeddings for a batch of texts using FastEmbed.
    """
    if isinstance(texts, (str, bytes)) or not hasattr(texts, "__iter__"):
        raise ValueError(
            "get_batch_fastembed_embedding expects a list of text values. "
            "Use get_fastembed_embedding(text) for one record, or pass [text1, text2, ...]."
        )

    cleaned = [_as_embedding_api_string(t) for t in texts]
    bad = [i for i, s in enumerate(cleaned) if not s]
    if bad:
        raise ValueError(
            f"Cannot embed {len(bad)} empty/NaN text(s) at indices {bad[:10]}{'...' if len(bad) > 10 else ''}. "
            "Filter those rows before batch embedding."
        )

    embedding_model = get_fastembed_model(model)
    embeddings: list[list[float]] = []
    for text_batch in batch_list(cleaned, batch_size):
        prefixed_batch = [_with_e5_prefix(text, input_type) for text in text_batch]
        embeddings.extend(vector.tolist() for vector in embedding_model.embed(prefixed_batch))

    return embeddings
