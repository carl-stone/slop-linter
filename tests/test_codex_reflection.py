import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import slop_linter as lint

try:
    from evals.codex import CodexLM
    from evals.optimize import MOCK_PROPOSAL, ReflectionLM
    from gepa.strategies.instruction_proposal import InstructionProposalSignature
except ImportError:
    CodexLM = None


@unittest.skipUnless(CodexLM is not None, "optional DSPy eval dependency not installed")
class CodexReflectionTests(unittest.TestCase):
    def test_standard_gepa_reflection_uses_chatgpt_login_and_jev_guidance(self):
        executions = []

        def run(command, **kwargs):
            self.assertTrue(all(key not in kwargs["env"] for key in
                                ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENROUTER_API_KEY")))
            if command[1:] == ["login", "status"]:
                return subprocess.CompletedProcess(command, 0, "", "Logged in using ChatGPT")
            executions.append((command, kwargs))
            self.assertEqual(list(Path(kwargs["cwd"]).iterdir()), [])
            return subprocess.CompletedProcess(command, 0,
                f"```\n{MOCK_PROPOSAL}\n```", "")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused-openai-key",
                                     "CODEX_API_KEY": "unused-codex-key",
                                     "OPENROUTER_API_KEY": "unused-openrouter-key"}), \
                patch("evals.codex.shutil.which", return_value="/fake/codex"), \
                patch("evals.codex.subprocess.run", side_effect=run):
            lm = ReflectionLM(model=None, api_key="unused-key")
            for _ in range(2):
                result = InstructionProposalSignature.run(lm=lambda prompt: lm(prompt)[0], input_dict={
                    "current_instruction_doc": "Is this a needless helper?",
                    "dataset_with_feedback": [{"source": "def alias(x): return original(x)",
                                                "feedback": "Expected flag: pure call alias."}]})
                self.assertEqual(result, {"new_instruction": MOCK_PROPOSAL})
        self.assertEqual(len(executions), 2)
        command, options = executions[0]
        self.assertNotIn("--model", command)  # Omission preserves the configured Codex model.
        request = options["input"]
        messages = json.loads(request.split("\n\n", 1)[1])
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Jev, a decision model", messages[0]["content"])
        self.assertIn("concise", messages[0]["content"])
        self.assertIn("Read the inputs carefully", messages[1]["content"])
        self.assertIn("pure call alias", messages[1]["content"])
        self.assertFalse(Path(options["cwd"]).exists())

    def test_missing_cli_or_chatgpt_login_fails_before_generation(self):
        with patch("evals.codex.shutil.which", return_value=None), \
                patch("evals.codex.subprocess.run") as run:
            with self.assertRaisesRegex(lint.LintError, "CLI on PATH"):
                CodexLM()
            run.assert_not_called()
        for status in (subprocess.CompletedProcess([], 1, "", "Not logged in"),
                       subprocess.CompletedProcess([], 0, "", "Logged in using an API key")):
            with self.subTest(status=status), \
                    patch("evals.codex.shutil.which", return_value="/fake/codex"), \
                    patch("evals.codex.subprocess.run", return_value=status) as run:
                with self.assertRaisesRegex(lint.LintError, "ChatGPT login"):
                    CodexLM()
                self.assertEqual(run.call_count, 1)

    def test_failed_empty_or_timed_out_generation_cannot_become_a_candidate(self):
        failures = [
            (subprocess.CompletedProcess([], 1, "", "model unavailable"), "model unavailable"),
            (subprocess.CompletedProcess([], 0, "", ""), "no response"),
            (subprocess.TimeoutExpired([], 120), "timed out"),
        ]
        for failure, message in failures:
            with self.subTest(message=message), \
                    patch("evals.codex.shutil.which", return_value="/fake/codex"), \
                    patch("evals.codex.subprocess.run", side_effect=[
                        subprocess.CompletedProcess([], 0, "", "Logged in using ChatGPT"), failure]):
                lm = CodexLM(model="chosen-model", timeout=120)
                with self.assertRaisesRegex(lint.LintError, message):
                    lm(messages=[{"role": "user", "content": "Propose a question."}])

    def test_openrouter_reflection_remains_explicit_and_does_not_start_codex(self):
        with patch("evals.codex.subprocess.run") as run, patch("evals.optimize.dspy.LM") as lm:
            lm.return_value.kwargs = {}
            lm.return_value.return_value = ["proposed question"]
            reflection = ReflectionLM(live=True, model="provider/model", api_key="test-key",
                                      backend="openrouter")
            self.assertEqual(reflection(messages=[]), ["proposed question"])
            self.assertEqual(lm.call_args.args[0], "openrouter/provider/model")
            self.assertEqual(lm.call_args.kwargs["api_key"], "test-key")
            run.assert_not_called()
            with self.assertRaisesRegex(lint.LintError, "needs --reflection-model"):
                ReflectionLM(live=True, model=None, api_key="test-key", backend="openrouter")


if __name__ == "__main__":
    unittest.main()
