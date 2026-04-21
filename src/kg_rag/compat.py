from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


try:
    from haystack import Document, component
except Exception:  # pragma: no cover - used in local tests without Haystack installed.

    @dataclass
    class Document:  # type: ignore[no-redef]
        content: str = ""
        meta: dict[str, Any] = field(default_factory=dict)
        id: str | None = None
        embedding: list[float] | None = None
        score: float | None = None

    class _ComponentCompat:
        def __call__(self, cls: type) -> type:
            return cls

        def output_types(self, **_types: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                return func

            return decorator

    component = _ComponentCompat()  # type: ignore[assignment]


def document_content(document: Document) -> str:
    return getattr(document, "content", "") or ""


def document_meta(document: Document) -> dict[str, Any]:
    meta = getattr(document, "meta", None)
    return dict(meta or {})


def document_id(document: Document) -> str | None:
    return getattr(document, "id", None)


def document_embedding(document: Document) -> list[float] | None:
    embedding = getattr(document, "embedding", None)
    return list(embedding) if embedding else None


def make_document(
    content: str,
    *,
    meta: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
    score: float | None = None,
    id: str | None = None,
) -> Document:
    kwargs: dict[str, Any] = {"content": content, "meta": meta or {}}
    if embedding is not None:
        kwargs["embedding"] = embedding
    if score is not None:
        kwargs["score"] = score
    if id is not None:
        kwargs["id"] = id
    return Document(**kwargs)
