"""Shared source records and user-facing errors."""

from dataclasses import dataclass


class LintError(Exception):
    """The scan could not be completed reliably."""


@dataclass(frozen=True)
class Function:
    path: str
    name: str
    line: int
    source: str
    column: int = 1
