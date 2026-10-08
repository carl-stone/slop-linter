"""Lint rule definitions shared by scanning and optimization."""

from dataclasses import dataclass
import json

from .models import LintError


QUESTION_ID = "SL001"
PREDICATE = (
    "Does this function merely rename or forward an existing operation or constructor, "
    "preset arguments, or shallowly assemble or override fields, filters, or options"
    "—including direct field replacement, redundant aliases, or stable callable identity—"
    "without adding validation, parsing, normalization, result handling, policy or "
    "mathematical constraint enforcement, data or shape transformation, a Python "
    "protocol (even by delegation), or required semantic adaptation such as converting "
    "structured inputs into the callee’s expected representation, extracting identifiers "
    "and assigning them to callback roles, or assembling a normalized specification "
    "(even with only a simple comprehension or returned mapping)?"
)


@dataclass(frozen=True)
class Rule:
    """A Noul rule: Yes means flag the behavior; No means it is acceptable.

    Prefer natural questions over awkward wording to force this convention.
    """

    code: str
    name: str
    predicate: str
    message: str
    demos: tuple[dict, ...] = ()


RULES = {
    "SL001": Rule("SL001", "needless-helper", PREDICATE, "needless forwarding helper"),
}


def check_instruction(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise LintError("decision instruction must be nonempty text")
    return instruction


def selected_rules(select: list[str], ignore: list[str]) -> list[Rule]:
    def resolve(names: list[str]) -> set[str]:
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise LintError("select and ignore must be lists of rule codes or names")
        aliases = {rule.name: rule.code for rule in RULES.values()}
        codes = set()
        for name in names:
            code = aliases.get(name, name)
            if code not in RULES:
                raise LintError(f"unknown rule {json.dumps(name)}; run slop-linter rules")
            codes.add(code)
        return codes
    enabled = resolve(select) - resolve(ignore)
    return [RULES[code] for code in sorted(enabled)]
