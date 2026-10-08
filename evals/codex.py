"""DSPy text generation through the user's authenticated Codex CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import dspy

import slop_linter as lint


class CodexLM(dspy.BaseLM):
    def __init__(self, model: str | None = None, timeout: float | None = None):
        super().__init__(model="codex/" + (model or "default"), cache=False)
        self.codex_model, self.timeout = model, timeout
        self.executable = shutil.which("codex")
        if not self.executable:
            raise lint.LintError("Codex reflection needs the codex CLI on PATH")
        # Use the saved ChatGPT login, even when API keys exist in the parent shell.
        self.env = {key: value for key, value in os.environ.items()
                    if key not in {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENROUTER_API_KEY"}}
        try:
            status = subprocess.run([self.executable, "login", "status"],
                                    env=self.env, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise lint.LintError("could not check Codex login; run codex login status") from exc
        if status.returncode or "chatgpt" not in (status.stdout + status.stderr).lower():
            raise lint.LintError("Codex reflection needs a ChatGPT login; run codex login")

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        if kwargs.get("n", 1) != 1:
            raise lint.LintError("Codex reflection generates one response per request")
        conversation = messages if messages is not None else [{"role": "user", "content": prompt}]
        request = (
            "Respond as the assistant to the conversation below. Follow its system instructions "
            "and requested output format exactly. Return only the requested response. "
            "Do not use tools, inspect files, or modify anything.\n\n"
            + json.dumps(conversation, ensure_ascii=False)
        )
        command = [self.executable, "exec", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--color", "never",
                   "-c", 'model_provider="openai"', "-c", "project_doc_max_bytes=0"]
        if self.codex_model:
            command += ["--model", self.codex_model]
        command.append("-")
        try:
            # Keep project instructions and repository files out of the reflection context.
            with tempfile.TemporaryDirectory(prefix="slop-reflection-") as folder:
                result = subprocess.run(command, input=request, cwd=folder, env=self.env,
                                        capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise lint.LintError(f"Codex reflection timed out after {self.timeout:g}s") from exc
        except OSError as exc:
            raise lint.LintError(f"could not run Codex reflection: {exc}") from exc
        if result.returncode:
            detail = result.stderr.strip()[-1000:] or f"exit status {result.returncode}"
            raise lint.LintError(f"Codex reflection failed: {detail}")
        response = result.stdout.strip()
        if not response:
            raise lint.LintError("Codex reflection returned no response")
        return [response]
