from __future__ import annotations

from typing import Any

from kg_rag.config import HuggingFaceConfig


def create_hf_chat_generator(config: HuggingFaceConfig) -> Any:
    config.require_runtime_values()
    from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
    from haystack.utils import Secret

    return HuggingFaceAPIChatGenerator(
        api_type="inference_endpoints",
        api_params={"url": config.endpoint_url},
        token=Secret.from_token(config.api_token),
    )


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
    text = str(reply)
    return text


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
