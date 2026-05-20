from functools import lru_cache

from app.core.config import settings


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        settings.hf_reranker_model_name,
        max_length=int(settings.hf_reranker_max_length),
    )


def score_hf_reranker(query: str, texts: list[str]) -> list[float]:
    if not settings.enable_hf_reranker:
        return []
    if not texts:
        return []

    try:
        model = _load_cross_encoder()
    except Exception:
        return []

    pairs = [(query, text) for text in texts]
    try:
        scores = model.predict(
            pairs,
            batch_size=max(1, int(settings.hf_reranker_batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception:
        return []

    return [float(score) for score in scores]
