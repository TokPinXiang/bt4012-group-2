"""LangChain chat model for the NUS SoC LLM service (SoCLaaS, OpenAI-compatible), with a disk cache.

Settings come from the SoCLaaS env file (default ~/.config/soclaas/soclaas.env, or set
SOCLAAS_ENV to another path). Environment variables of the same name override it.
The key is never logged or written to outputs.

The model name comes from the agent definition (definitions/agents/*.md). qwen3.6:35b is
pinned there rather than the gateway's moving `default` alias, so results are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.messages import AIMessage, BaseMessage, message_to_dict, messages_from_dict, messages_to_dict
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_ENV = Path.home() / ".config" / "soclaas" / "soclaas.env"


def soclaas_settings(env_path: Path | None = None) -> dict:
    path = Path(env_path or os.environ.get("SOCLAAS_ENV") or DEFAULT_ENV)
    cfg = dotenv_values(path) if path.exists() else {}
    out = {k: os.environ.get(k) or cfg.get(k) for k in ("SOCLAAS_BASE_URL", "SOCLAAS_API_KEY")}
    if not out["SOCLAAS_API_KEY"] or not out["SOCLAAS_BASE_URL"]:
        raise RuntimeError(f"SOCLAAS_BASE_URL / SOCLAAS_API_KEY not found in {path} or the environment")
    return out


def make_chat_model(model: str, temperature: float = 0.0, max_tokens: int = 4000, timeout: float = 300,
                    env_path: Path | None = None):
    from langchain_openai import ChatOpenAI

    s = soclaas_settings(env_path)
    return ChatOpenAI(model=model, base_url=s["SOCLAAS_BASE_URL"], api_key=s["SOCLAAS_API_KEY"],
                      temperature=temperature, max_tokens=max_tokens, timeout=timeout, max_retries=4)


class AgentLLM:
    """Wraps any LangChain chat model that supports bind_tools; identical calls hit the disk cache."""

    def __init__(self, chat_model, model_name: str, cache_dir: Path | None = None):
        self.chat_model = chat_model
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) / model_name.replace(":", "_") if cache_dir else None

    def invoke(self, messages: list[BaseMessage], tools: list, tool_choice: str | None = None) -> AIMessage:
        cache_file = None
        if self.cache_dir:
            key = json.dumps({"model": self.model_name, "messages": messages_to_dict(messages),
                              "tools": [convert_to_openai_tool(t) for t in tools], "tool_choice": tool_choice},
                             sort_keys=True, default=str)
            cache_file = self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"
            if cache_file.exists():
                return messages_from_dict([json.loads(cache_file.read_text(encoding="utf-8"))])[0]

        bound = self.chat_model.bind_tools(tools, tool_choice=tool_choice) if tool_choice else self.chat_model.bind_tools(tools)
        reply = bound.invoke(messages)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(message_to_dict(reply), default=str), encoding="utf-8")
        return reply
