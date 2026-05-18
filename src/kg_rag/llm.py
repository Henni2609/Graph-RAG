from __future__ import annotations

import functools
from typing import Any, AsyncGenerator

from kg_rag.config import LLMConfig


@functools.lru_cache(maxsize=None)
def _cached_chat_generator(
    config: LLMConfig,
    model: str | None,
    timeout: float | None,
    max_retries: int | None,
) -> Any:
    config.require_runtime_values()
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    extra: dict[str, Any] = {}
    if timeout is not None:
        extra["timeout"] = timeout
    if max_retries is not None:
        extra["max_retries"] = max_retries

    return OpenAIChatGenerator(
        api_key=Secret.from_token(config.api_key),
        model=model or config.model,
        api_base_url=config.base_url,
        **extra,
    )


def create_chat_generator(
    config: LLMConfig,
    *,
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Any:
    return _cached_chat_generator(config, model, timeout, max_retries)


def make_chat_messages(system_prompt: str, user_prompt: str) -> list[Any]:
    from haystack.dataclasses import ChatMessage

    return [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(user_prompt),
    ]


def chat_reply_text(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    for attr in ("text", "content"):
        value = getattr(reply, attr, None)
        if isinstance(value, str):
            return value
    return str(reply)


def run_chat(
    generator: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    generation_kwargs: dict[str, Any] | None = None,
) -> str:
    result = generator.run(
        messages=make_chat_messages(system_prompt, user_prompt),
        generation_kwargs=generation_kwargs,
    )
    replies = result.get("replies", [])
    if not replies:
        return ""
    return chat_reply_text(replies[0])


@functools.lru_cache(maxsize=None)
def _get_async_openai_client(api_key: str, base_url: str) -> Any:
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def stream_chat_tokens(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    generation_kwargs: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    config.require_runtime_values()
    client = _get_async_openai_client(config.api_key, config.base_url)

    gk = dict(generation_kwargs or {})
    timeout = gk.pop("timeout", 60)

    stream = await client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        timeout=timeout,
        **gk,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
