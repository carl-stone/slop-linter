import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import slop_linter as lint


def response(probability=0.1, **updates):
    result = {"model": "typesafe/jev-1.13-20260917",
              "answers": {lint.QUESTION_ID: {"type": "noul", "noul": probability}},
              "usage": {"input_tokens": 50, "output_tokens": 0, "cost": 0}}
    result.update(updates)
    return io.BytesIO(json.dumps(result).encode())


class LocalTest(unittest.TestCase):
    def setUp(self):
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        # All HTTP is mocked. An unconfigured request fails the test immediately.
        self.http = self.contexts.enter_context(patch(
            "slop_linter.transport.urllib.request.OpenerDirector.open",
            side_effect=AssertionError("unexpected HTTP request"),
        ))


class ExtractionTests(LocalTest):
    def test_methods_async_nested_and_multiline_decorators_keep_formatting(self):
        source = (
            "raise RuntimeError('never execute scanned modules')\r\n"
            "class Client:\r\n"
            "    @(\r\n"
            "        decorate\r\n"
            "    )\r\n"
            "    @other(option='é')\r\n"
            "    async def send(self, value):  # keep this comment\r\n"
            "        def inner():\r\n"
            "            return value\r\n"
            "        return inner()\r\n"
            "\r\n"
            "def after(): return 'a\x85b'  # keep trailing comment\r\n"
        )
        functions = lint.extract_functions("client.py", source.encode())
        self.assertEqual([(f.name, f.line) for f in functions],
                         [("send", 7), ("inner", 8), ("after", 12)])
        self.assertEqual(functions[0].source,
                         "    @(\r\n        decorate\r\n    )\r\n"
                         "    @other(option='é')\r\n"
                         "    async def send(self, value):  # keep this comment\r\n"
                         "        def inner():\r\n            return value\r\n"
                         "        return inner()\r\n")
        self.assertEqual(functions[1].source,
                         "        def inner():\r\n            return value\r\n")
        self.assertEqual(functions[2].source,
                         "def after(): return 'a\x85b'  # keep trailing comment\r\n")
        self.http.assert_not_called()

    def test_encoding_cookie_and_stacked_decorators(self):
        source = ("# coding: latin-1\n@first\n# comment\n@second(\n    'é'\n)\n"
                  "def café():\n    return 'é'\n")
        function, = lint.extract_functions("latin.py", source.encode("latin-1"))
        self.assertEqual(function.name, "café")
        self.assertEqual(function.line, 7)
        self.assertEqual(function.source, source[source.index("@first"):])

    def test_invalid_source_is_an_error(self):
        for blob in (b"def broken(:\n", b"# coding: unknown-codec\ndef f(): pass\n",
                     b"def f(): return '\xff'\n", b"def f():\0pass\n"):
            with self.subTest(blob=blob), self.assertRaises(lint.LintError):
                lint.extract_functions("bad.py", blob)


class RepositoryTests(LocalTest):
    def setUp(self):
        super().setUp()
        self.repo = Path(self.contexts.enter_context(tempfile.TemporaryDirectory()))
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("GIT_") and key != "OPENROUTER_API_KEY"}
        environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.contexts.enter_context(patch.dict(os.environ, environment, clear=True))
        self.run_git("init", "--quiet")
        self.run_git("config", "user.email", "test@example.invalid")
        self.run_git("config", "user.name", "Local Test")
        self.run_git("config", "core.hooksPath", os.devnull)
        self.run_git("config", "core.autocrlf", "false")

    def run_git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def stage(self, name, source):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        self.run_git("add", "--", name)
        return path

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = lint.main(["check", "--staged", "--repo", str(self.repo), *args])
        return code, out.getvalue(), err.getvalue()

    def test_index_contents_win_over_unstaged_changes_even_from_subdirectory(self):
        file = self.stage("pkg/app.py", "def staged():\n    return target()\n")
        file.write_text("not even valid Python: unstaged changes\n")
        (self.repo / "untracked.py").write_text("def untracked(): pass\n")
        functions = lint.staged_functions(self.repo / "pkg")
        self.assertEqual([f.name for f in functions], ["staged"])
        self.assertEqual(functions[0].path, "pkg/app.py")
        self.assertEqual(functions[0].source, "def staged():\n    return target()\n")

    def test_unstaged_functions_are_not_selected(self):
        file = self.stage("unchanged.py", "def original(): pass\n")
        self.run_git("commit", "--quiet", "-m", "baseline")
        file.write_text("def unstaged(): pass\n")
        self.assertEqual(lint.staged_functions(self.repo), [])

    def test_weird_paths_are_lossless_and_diagnostics_escape_controls(self):
        names = ["space name.py", "-leading.py", "colon:tab\tline\n雪.py"]
        for name in names:
            self.stage(name, "def f(): return target()\n")
        self.assertEqual({f.path for f in lint.staged_functions(self.repo)}, set(names))
        self.http.side_effect = lambda *args, **kwargs: response(0.9)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            code, out, err = self.cli()
        self.assertEqual(code, 1)
        self.assertFalse(err)
        self.assertIn(json.dumps(names[2]) + ":1:", out)
        self.assertEqual(len(out.splitlines()), 4)

    def test_deleted_and_renamed_files(self):
        self.stage("deleted.py", "def deleted(): pass\n")
        self.stage("old.py", "def renamed(): pass\n")
        self.run_git("commit", "--quiet", "-m", "baseline")
        self.run_git("rm", "--quiet", "--", "deleted.py")
        self.run_git("mv", "--", "old.py", "new.py")
        functions = lint.staged_functions(self.repo)
        self.assertEqual([(f.path, f.name) for f in functions], [("new.py", "renamed")])

    def test_staged_symlink_and_unmerged_index_are_errors(self):
        (self.repo / "link.py").symlink_to("somewhere.py")
        self.run_git("add", "--", "link.py")
        with self.assertRaisesRegex(lint.LintError, "not a regular file"):
            lint.staged_functions(self.repo)
        self.run_git("rm", "--cached", "--quiet", "--", "link.py")
        self.stage("conflict.py", "def f(): pass\n")
        oid = self.run_git("rev-parse", ":conflict.py").strip().decode()
        self.run_git("update-index", "--force-remove", "--", "conflict.py")
        subprocess.run(["git", "-C", str(self.repo), "update-index", "--index-info"],
                       input=f"100644 {oid} 1\tconflict.py\n100644 {oid} 2\tconflict.py\n".encode(),
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self.assertRaisesRegex(lint.LintError, "unresolved"):
            lint.staged_functions(self.repo)

    def test_clean_flag_threshold_and_error_exit_codes_with_mocked_http(self):
        self.stage("app.py", "def f(): return target()\ndef g(): return target()\n")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            for values, expected in (([0.1, 0.49], 0), ([0.1, 0.5], 1)):
                self.http.side_effect = [response(value) for value in values]
                code, out, err = self.cli()
                self.assertEqual(code, expected)
                self.assertIn("checked 2 functions", out)
                self.assertFalse(err)
            self.http.side_effect = lambda *args, **kwargs: response(0.8)
            self.assertEqual(self.cli("--threshold", "0.9")[0], 0)
            self.http.side_effect = [response(0.9), urllib.error.URLError("fake-test-key")]
            code, out, err = self.cli()
        self.assertEqual(code, 2)  # An error takes precedence over an earlier violation.
        self.assertIn("SL001", out)
        self.assertIn('app.py:2:1:', err)
        self.assertNotIn("fake-test-key", err)

    def test_missing_key_dry_run_empty_and_syntax_error_do_not_call_http(self):
        self.assertEqual(self.cli()[0], 0)
        self.stage("app.py", "def f(): pass\n")
        self.assertEqual(self.cli()[0], 2)
        code, out, err = self.cli("--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)[0]["source"], "def f(): pass\n")
        self.assertFalse(err)
        self.stage("broken.py", "def invalid(:\n")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            self.assertEqual(self.cli()[0], 2)
        self.http.assert_not_called()

    def test_not_a_repository_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(lint.LintError):
                lint.staged_functions(Path(folder))

    def test_staged_exclusion_skips_invalid_source_before_parsing(self):
        self.stage("generated/bad.py", "not Python :\n")
        self.stage("app.py", "def f(): pass\n")
        functions = lint.staged_functions(self.repo, ["generated/*"])
        self.assertEqual([function.path for function in functions], ["app.py"])
        self.http.assert_not_called()


class HttpTests(LocalTest):
    function = lint.Function("app.py", "f", 1, "def f(): return target()\n")

    def test_exact_openrouter_request(self):
        self.http.side_effect = None
        self.http.return_value = response(0.75)
        probability, model = lint.judge(self.function, "fake-test-key", "chosen-model", 7)
        request = self.http.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-test-key")
        self.assertEqual(self.http.call_args.kwargs["timeout"], 7)
        self.assertEqual(json.loads(request.data), {
            "model": "chosen-model", "state": self.function.source,
            "questions": {"SL001": {"type": "noul", "instructions": lint.PREDICATE}},
        })
        self.assertEqual((probability, model), (0.75, "typesafe/jev-1.13-20260917"))

    def test_malformed_probabilities_types_and_shapes_fail(self):
        cases = [response(value) for value in (True, False, None, "0.9", -0.1, 1.1,
                                               float("nan"), float("inf"), -float("inf"))]
        cases += [response(answers={}), response(answers={lint.QUESTION_ID: {"type": "choice", "noul": 0.9}}),
                  response(answers=[]), response(answers={lint.QUESTION_ID: None}),
                  response(model=""), response(model=None), response(error={"message": "failure"}),
                  io.BytesIO(b"[]"), io.BytesIO(b"not JSON"), io.BytesIO(b"\xff"),
                  io.BytesIO(b" " * (lint.MAX_RESPONSE_BYTES + 1))]
        for index, item in enumerate(cases):
            with self.subTest(case=index):
                self.http.side_effect = None
                self.http.return_value = item
                with self.assertRaises(lint.LintError):
                    lint.judge(self.function, "fake-test-key", lint.DEFAULT_MODEL, 30)

    def test_http_error_timeout_and_redirect_fail_without_leaking_details(self):
        cases = [urllib.error.HTTPError(lint.API_URL, 401, "fake-test-key", {},
                                      io.BytesIO(b"secret response body")),
                 urllib.error.URLError("fake-test-key"), TimeoutError("fake-test-key")]
        for error in cases:
            with self.subTest(error=type(error).__name__):
                self.http.side_effect = error
                with self.assertRaises(lint.LintError) as caught:
                    lint.judge(self.function, "fake-test-key", lint.DEFAULT_MODEL, 30)
                self.assertNotIn("fake-test-key", str(caught.exception))
                self.assertNotIn("secret response body", str(caught.exception))
        self.assertIsNone(lint._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid"))

    def test_invalid_configuration_exits_two_without_http(self):
        for args in (["--threshold", "nan"], ["--threshold", "1.1"],
                     ["--timeout", "0"], ["--timeout", "inf"], ["--model", ""]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(lint.main(["check", *args]), 2)
        self.http.assert_not_called()


class PathAndConfigTests(LocalTest):
    def setUp(self):
        super().setUp()
        self.root = Path(self.contexts.enter_context(tempfile.TemporaryDirectory()))
        self.contexts.enter_context(patch.dict(os.environ,
            {key: value for key, value in os.environ.items() if key != "OPENROUTER_API_KEY"},
            clear=True))

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def cli(self, *args, cwd=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = lint.main(["check", "--repo", str(cwd or self.root), *args])
        return code, out.getvalue(), err.getvalue()

    def test_paths_recurse_deduplicate_and_exclude_before_parsing(self):
        self.write("src/app.py", "def actual():\n    return 42\n")
        self.write("src/generated/bad.py", "not Python :\n")
        self.write(".venv/bad.py", "also not Python :\n")
        self.write(".venv-eval/bad.py", "also not Python :\n")
        self.write("notes.txt", "def not_python(): pass\n")
        code, out, err = self.cli(".", "src/app.py", "--exclude", "generated", "--dry-run")
        self.assertEqual(code, 0)
        self.assertFalse(err)
        functions = json.loads(out)
        self.assertEqual([(f["path"], f["name"]) for f in functions], [("src/app.py", "actual")])
        self.http.assert_not_called()

    def test_config_discovery_model_threshold_and_cli_precedence(self):
        self.write("pyproject.toml", '[tool.slop-linter]\nselect = ["needless-helper"]\n'
                   'model = "configured-model"\nthreshold = 0.9\ntimeout = 12\n')
        child = self.write("src/app.py", "def f(): return target()\n").parent
        self.http.side_effect = lambda *args, **kwargs: response(0.8)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            self.assertEqual(self.cli("app.py", cwd=child)[0], 0)
            request = self.http.call_args.args[0]
            self.assertEqual(json.loads(request.data)["model"], "configured-model")
            self.assertEqual(self.http.call_args.kwargs["timeout"], 12)
            self.assertEqual(self.cli("app.py", "--threshold", "0.7", "--model", "cli-model", cwd=child)[0], 1)
            self.assertEqual(json.loads(self.http.call_args.args[0].data)["model"], "cli-model")
            self.assertEqual(self.cli("app.py", "--no-config", cwd=child)[0], 1)

    def test_explicit_config_exclusions_and_selection_ignore(self):
        self.write("settings.toml", '[tool.slop-linter]\nselect = ["SL001"]\n'
                   'ignore = ["SL001"]\nexclude = ["generated/*"]\n')
        self.write("app.py", "def f(): return target()\n")
        self.write("generated/bad.py", "not valid Python :\n")
        self.assertEqual(self.cli("--config", "settings.toml")[0], 0)
        self.http.assert_not_called()
        self.http.side_effect = lambda *args, **kwargs: response(0.9)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            code, out, err = self.cli("--config", "settings.toml", "--ignore", "")
        self.assertEqual(code, 1)
        self.assertIn("app.py:1:1: SL001 needless-helper:", out)
        self.assertFalse(err)
        self.assertEqual(self.http.call_count, 1)

    def test_unknown_rules_and_invalid_config_are_errors_without_http(self):
        for args in (["--select", "SL999"], ["--ignore", "typo"], ["--select", ""]):
            with self.subTest(args=args):
                self.assertEqual(self.cli(*args)[0], 0 if args[-1] == "" else 2)
        for text in ('[tool.slop-linter]\nthreshold = true\n',
                     '[tool.slop-linter]\nignore = "SL001"\n',
                     '[tool.slop-linter]\nexclude = [false]\n',
                     '[tool.slop-linter]\nunknown = 1\n',
                     '[tool.slop-linter]\ntimeout = inf\n',
                     'tool = 1\n', '[broken', '[tool.other]\nx = 1\n'):
            with self.subTest(text=text):
                self.write("settings.toml", text)
                self.assertEqual(self.cli("--config", "settings.toml")[0], 2)
        self.assertEqual(self.cli("--config", "missing.toml")[0], 2)
        self.http.assert_not_called()

    def test_missing_path_and_symlink_are_errors(self):
        self.assertEqual(self.cli("missing.py", "--dry-run")[0], 2)
        real = self.write("app.py", "def f(): pass\n")
        (self.root / "link.py").symlink_to(real)
        self.assertEqual(self.cli("link.py", "--dry-run")[0], 2)
        self.http.assert_not_called()

    def test_rule_registry_runs_a_new_predicate_and_message(self):
        self.write("app.py", "def f(): return target()\n")
        rule = lint.Rule("TEST001", "test-rule", "Does this test condition hold?", "test diagnostic")
        self.http.side_effect = lambda *args, **kwargs: response(
            answers={"TEST001": {"type": "noul", "noul": 0.9}})
        with patch.dict(lint.RULES, {"TEST001": rule}, clear=True), patch.dict(
                os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            self.assertEqual(lint.load_config(self.root, None, True)["select"], ["TEST001"])
            code, out, err = self.cli("app.py", "--select", "TEST001")
        self.assertEqual(code, 1)
        self.assertIn("TEST001 test-rule:", out)
        self.assertIn("test diagnostic", out)
        self.assertFalse(err)
        self.assertEqual(json.loads(self.http.call_args.args[0].data)["questions"],
                         {"TEST001": {"type": "noul", "instructions": rule.predicate}})

    def test_optimized_candidate_drives_the_actual_request_without_mutating_rule(self):
        self.write("app.py", "def f(): return target()\n")
        prompt = {"rule": "SL001", "instruction": "Is this only a redundant call alias?",
                  "model": "candidate-model", "threshold": 0.95, "timeout": 11,
                  "demos": [{"source": "def alias(x): return target(x)\n", "label": True}]}
        self.write("optimized.json", json.dumps({"schema_version": 1, "candidate": prompt}))
        original = lint.RULES["SL001"]
        self.http.side_effect = lambda *args, **kwargs: response(0.9)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            self.assertEqual(self.cli("app.py", "--candidate", "optimized.json")[0], 0)
            payload = json.loads(self.http.call_args.args[0].data)
            self.assertEqual(payload["questions"]["SL001"]["instructions"], prompt["instruction"])
            self.assertEqual(payload["model"], "candidate-model")
            self.assertEqual(payload["state"], {
                "instructions": "Apply the question to inputs.source. The demos are labeled examples "
                                "of the same question: true means Yes and false means No.",
                "input_fields": "source: Python function source to assess.",
                "inputs": {"source": "def f(): return target()\n"},
                "demos": [{"source": prompt["demos"][0]["source"], "SL001": True}]})
            self.assertEqual(self.http.call_args.kwargs["timeout"], 11)
            self.assertEqual(self.cli("app.py", "--candidate", "optimized.json",
                                      "--threshold", "0.8", "--model", "cli-model")[0], 1)
            self.assertEqual(json.loads(self.http.call_args.args[0].data)["model"], "cli-model")
        self.assertIs(lint.RULES["SL001"], original)

    def test_config_candidate_path_is_relative_to_the_configuration_file(self):
        prompt = {"rule": "SL001", "instruction": "Is this only a redundant call alias?",
                  "model": "candidate-model", "threshold": 0.8, "timeout": 9}
        self.write("models/optimized.json", json.dumps(prompt))
        self.write("pyproject.toml", '[tool.slop-linter]\ncandidate = "models/optimized.json"\n')
        child = self.write("src/app.py", "def f(): return target()\n").parent
        self.http.side_effect = lambda *args, **kwargs: response(0.9)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key"}):
            self.assertEqual(self.cli("app.py", cwd=child)[0], 1)
        self.assertEqual(json.loads(self.http.call_args.args[0].data)["questions"]["SL001"],
                         {"type": "noul", "instructions": prompt["instruction"]})

    def test_bad_or_inactive_candidate_is_rejected_before_scanning_or_requesting(self):
        prompt = {"rule": "SL001", "instruction": "Is this only a redundant call alias?",
                  "model": "candidate-model", "threshold": 0.8, "timeout": 9}
        invalid = [[], {"schema_version": 2, "candidate": prompt},
                   {**prompt, "rule": []}, {**prompt, "rule": "SL999"},
                   {**prompt, "instruction": " "},
                   {**prompt, "model": " "}, {**prompt, "threshold": True},
                   {**prompt, "threshold": float("nan")}, {**prompt, "timeout": 0},
                   {**prompt, "demos": [{"source": "def f(): pass", "label": "Yes"}]},
                   {key: value for key, value in prompt.items() if key != "instruction"}]
        for artifact in invalid:
            with self.subTest(artifact=artifact):
                self.write("optimized.json", json.dumps(artifact))
                self.assertEqual(self.cli("--candidate", "optimized.json", "--dry-run")[0], 2)
        self.write("optimized.json", json.dumps(prompt))
        self.assertEqual(self.cli("--candidate", "optimized.json", "--ignore", "SL001")[0], 2)
        self.http.assert_not_called()

    def test_rules_command_and_staged_path_conflict_do_not_use_http(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(lint.main(["rules"]), 0)
        self.assertIn("SL001 needless-helper", output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.cli("app.py", "--staged")
        self.assertEqual(caught.exception.code, 2)
        self.http.assert_not_called()

    def test_eval_command_dispatches_unchanged_arguments(self):
        with patch("evals.cli.main", return_value=0) as evaluate:
            self.assertEqual(lint.main(["eval", "--data", "examples", "validate"]), 0)
        evaluate.assert_called_once_with(["--data", "examples", "validate"])


if __name__ == "__main__":
    unittest.main()
