from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def get_model():
    """The only place the embedding model is loaded. Cached after first call."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts):
    """ Embed a list of texts. Returns normalized embeddings as plain lists."""
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def count_tokens(text):
    """Token length using the model's own tokenizer (includes [CLS]/[SEP])."""
    return len(get_model().tokenizer(text)["input_ids"])
