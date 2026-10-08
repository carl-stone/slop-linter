"""Rule-based Python linting, with optional DSPy rule optimization."""

from .cli import main
from .config import (DEFAULT_THRESHOLD, DEFAULT_TIMEOUT, load_candidate,
                     load_config, validate_prompt)
from .extraction import (DEFAULT_EXCLUDES, excluded, extract_functions, git,
                         path_functions, staged_functions)
from .models import Function, LintError
from .rules import PREDICATE, QUESTION_ID, RULES, Rule, check_instruction, selected_rules
from .transport import API_URL, DEFAULT_MODEL, MAX_RESPONSE_BYTES, _NoRedirect, judge

__all__ = [
    "Function", "LintError", "RULES", "Rule", "check_instruction", "extract_functions",
    "judge", "load_candidate", "load_config", "main", "path_functions", "selected_rules",
    "staged_functions", "validate_prompt",
]
