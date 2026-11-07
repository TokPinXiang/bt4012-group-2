"""Load agent and tool definitions from Markdown files with YAML frontmatter.

    definitions/agents/<name>.md   frontmatter = settings, body = system prompt
    definitions/tools/<name>.md    frontmatter = name, description, handler, args; body = usage notes

Specs are validated with Pydantic when loaded, and cross-checked: every tool an agent lists
must exist, and every tool's handler must exist in the toolkit. Edit the .md files to change
prompts or tool descriptions without touching Python.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from . import schemas

DEFINITIONS_DIR = Path(__file__).parent / "definitions"
NAME = r"^[a-z][a-z0-9_]*$"
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(text.lstrip("﻿"))
    if not m:
        raise ValueError("file must start with a '---' YAML frontmatter block")
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return meta, m.group(2).strip()


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolArg(_Spec):
    type: Literal["string", "integer", "number", "boolean"]
    description: str
    required: bool = True
    default: Any = None
    pattern: str | None = None
    minimum: float | None = None
    maximum: float | None = None


class ToolSpec(_Spec):
    name: str = Field(pattern=NAME)
    description: str = Field(min_length=1)
    handler: str | None = Field(default=None, pattern=NAME, description="CaseToolkit method that runs the tool")
    args: dict[str, ToolArg] = {}
    args_schema: str | None = Field(default=None, description="Name of a Pydantic model in schemas.py")
    notes: str = ""
    source: str = ""

    @model_validator(mode="after")
    def _one_schema(self):
        if self.args and self.args_schema:
            raise ValueError("give either args or args_schema, not both")
        if self.args_schema and not isinstance(getattr(schemas, self.args_schema, None), type):
            raise ValueError(f"args_schema {self.args_schema!r} is not a model in schemas.py")
        if not self.args_schema and not self.handler:
            raise ValueError("a tool without args_schema needs a handler")
        return self

    @property
    def full_description(self) -> str:
        return f"{self.description}\n\n{self.notes}".strip()

    def args_model(self) -> type[BaseModel]:
        if self.args_schema:
            return getattr(schemas, self.args_schema)
        py = {"string": str, "integer": int, "number": float, "boolean": bool}
        fields = {}
        for name, a in self.args.items():
            constraints = {k: v for k, v in (("pattern", a.pattern), ("ge", a.minimum), ("le", a.maximum)) if v is not None}
            default = ... if a.required else a.default
            fields[name] = (py[a.type] if a.required else py[a.type] | None, Field(default, description=a.description, **constraints))
        return create_model(f"{self.name.title().replace('_', '')}Args", __config__=ConfigDict(extra="forbid"), **fields)


class AgentSpec(_Spec):
    name: str = Field(pattern=NAME)
    description: str
    model: str
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(4000, ge=256)
    max_tool_calls: int = Field(4, ge=0, le=20)
    max_retries: int = Field(1, ge=0, le=5)
    tools: list[str] = []
    submit_tool: str
    prompt: str = Field(min_length=1)
    source: str = ""


def _load(path: Path, cls, body_field: str):
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    try:
        return cls.model_validate({**meta, body_field: body, "source": str(path)})
    except Exception as exc:
        raise ValueError(f"invalid definition {path.name}: {exc}") from exc


def load_tool_specs(directory: Path = DEFINITIONS_DIR / "tools") -> dict[str, ToolSpec]:
    specs = {}
    for path in sorted(Path(directory).glob("*.md")):
        spec = _load(path, ToolSpec, "notes")
        if spec.name != path.stem:
            raise ValueError(f"{path.name}: name {spec.name!r} must match the file name")
        specs[spec.name] = spec
    return specs


def load_agent_spec(name: str = "investigator", directory: Path = DEFINITIONS_DIR,
                    toolkit_cls: type | None = None) -> tuple[AgentSpec, dict[str, ToolSpec]]:
    """Load one agent plus the tools it uses, and check that everything it refers to exists."""
    agent = _load(Path(directory) / "agents" / f"{name}.md", AgentSpec, "prompt")
    tools = load_tool_specs(Path(directory) / "tools")
    wanted = agent.tools + [agent.submit_tool]
    missing = [t for t in wanted if t not in tools]
    if missing:
        raise ValueError(f"agent {agent.name!r} lists tools with no definition: {missing}")
    if not tools[agent.submit_tool].args_schema:
        raise ValueError(f"submit tool {agent.submit_tool!r} must declare args_schema")
    if toolkit_cls is not None:
        bad = [t for t in agent.tools if not callable(getattr(toolkit_cls, tools[t].handler or "", None))]
        if bad:
            raise ValueError(f"tools {bad} point to handlers missing from {toolkit_cls.__name__}")
    return agent, {t: tools[t] for t in wanted}
