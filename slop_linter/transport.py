"""OpenRouter Decisions transport used by linting and DSPy."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

from .models import Function, LintError
from .rules import RULES, Rule


API_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
MAX_RESPONSE_BYTES = 1_048_576
DEMO_CONTEXT = (
    "Apply the question to inputs.source. The demos are labeled examples of the same question: "
    "true means Yes and false means No."
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the API key or source to a redirected endpoint.
        return None


def judge(function: Function, api_key: str, model: str, timeout: float,
          rule: Rule = RULES["SL001"]) -> tuple[float, str]:
    payload = {
        "model": model,
        "state": ({"instructions": DEMO_CONTEXT,
                   "input_fields": "source: Python function source to assess.",
                   "inputs": {"source": function.source},
                   "demos": [{"source": demo["source"], rule.code: demo["label"]}
                             for demo in rule.demos]}
                  if rule.demos else function.source),
        "questions": {rule.code: {"type": "noul", "instructions": rule.predicate}},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # Do not print the response body: it can echo credentials or source.
        exc.close()
        raise LintError(f"OpenRouter returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        raise LintError("OpenRouter request failed or timed out") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise LintError("OpenRouter response exceeds 1 MiB")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise LintError("OpenRouter returned invalid JSON") from exc
    if not isinstance(result, dict) or "error" in result:
        raise LintError("OpenRouter returned an error or invalid response")
    answers = result.get("answers")
    if not isinstance(answers, dict) or set(answers) != {rule.code}:
        raise LintError("OpenRouter returned unexpected answer keys")
    answer = answers[rule.code]
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise LintError("OpenRouter returned an invalid Noul answer")
    probability = answer.get("noul")
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        raise LintError("OpenRouter Noul probability must be a finite number in [0, 1]")
    resolved_model = result.get("model")
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise LintError("OpenRouter response is missing its model")
    return float(probability), resolved_model
