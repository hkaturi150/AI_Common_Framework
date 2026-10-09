"""
Prompt template management — version, render, and test prompts.

Usage:
    tmpl = PromptTemplate(
        name="summarise",
        template="Summarise the following in {style} style:\n\n{text}",
    )
    rendered = tmpl.render(style="bullet points", text="...")
    msg = tmpl.to_message()  # returns Message(role=USER, content=rendered)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_core.llm.types import Message, Role


@dataclass
class PromptTemplate:
    name: str
    template: str
    version: str = "1.0"
    role: Role = Role.USER
    description: str = ""
    tags: List[str] = field(default_factory=list)
    _rendered: Optional[str] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, **kwargs: Any) -> str:
        """Render template with provided variables (simple {var} substitution)."""
        missing = self.variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"PromptTemplate '{self.name}' missing variables: {missing}")
        self._rendered = self.template.format(**kwargs)
        return self._rendered

    def to_message(self, **kwargs: Any) -> Message:
        """Render and return as a Message object."""
        return Message(role=self.role, content=self.render(**kwargs))

    @property
    def variables(self) -> set:
        """Return set of {variable} names in the template."""
        return set(re.findall(r"\{(\w+)\}", self.template))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "role": self.role.value,
            "description": self.description,
            "tags": self.tags,
            "template": self.template,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PromptTemplate":
        return cls(
            name=d["name"],
            template=d["template"],
            version=d.get("version", "1.0"),
            role=Role(d.get("role", "user")),
            description=d.get("description", ""),
            tags=d.get("tags", []),
        )

    def save(self, directory: str | Path) -> Path:
        p = Path(directory) / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PromptTemplate":
        return cls.from_dict(json.loads(Path(path).read_text()))


class PromptLibrary:
    """
    In-memory registry for multiple PromptTemplates.
    Load once, use everywhere.
    """

    def __init__(self):
        self._store: Dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        self._store[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        if name not in self._store:
            raise KeyError(f"No prompt template named '{name}'")
        return self._store[name]

    def load_directory(self, directory: str | Path) -> int:
        """Load all .json prompt files from a directory. Returns count loaded."""
        count = 0
        for p in Path(directory).glob("*.json"):
            self.register(PromptTemplate.load(p))
            count += 1
        return count

    def render(self, name: str, **kwargs: Any) -> str:
        return self.get(name).render(**kwargs)

    def to_message(self, name: str, **kwargs: Any) -> Message:
        return self.get(name).to_message(**kwargs)

    @property
    def names(self) -> List[str]:
        return list(self._store.keys())
