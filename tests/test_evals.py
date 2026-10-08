import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from evals import core
from evals.cli import main
from evals.extract import repository_functions
import slop_linter as lint

try:
    from evals.optimize import DecisionProgram, LabeledMIPRO, MOCK_PROPOSAL, optimize
    import dspy
except ImportError:
    dspy = None


DATA = core.DATA_ROOT / "SL001" / "v1"


class EvaluationTests(unittest.TestCase):
    def test_repository_extraction_keeps_methods_and_nested_functions_without_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.py").write_text(
                "raise RuntimeError('do not execute')\n"
                "class Client:\n"
                "    @decorator\n"
                "    def send(self, value):\n"
                "        def prepare(item):\n"
                "            return item.strip()\n"
                "        return self.backend.send(prepare(value))\n")
            environment = root / ".venv-eval"
            environment.mkdir()
            (environment / "dependency.py").write_text("def ignored(): pass\n")
            rows = repository_functions(root)
            self.assertEqual([row["qualname"] for row in rows], ["Client.send", "Client.send.prepare"])
            self.assertTrue(rows[0]["source"].startswith("@decorator\ndef send"))
            self.assertIn("def prepare", rows[0]["source"])

    def test_real_nested_function_examples_need_live_predictions_not_fake_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            source = "def outer(value):\n    def clean(item):\n        return item.strip()\n    return clean(value)\n"
            row = {"id": "nested", "group": "nested", "split": "train", "label": False,
                   "rationale": "Normalizes the value.", "source": source}
            path = data / "train.jsonl"
            path.write_text(json.dumps(row) + "\n")
            metadata = {"schema_version": 1, "rule": "SL001", "version": "nested-test",
                        "files": {"train": {"count": 1, "sha256": core.digest(path)}}}
            (data / "manifest.json").write_text(json.dumps(metadata))
            cases = core.load_split(data, "train")
            self.assertEqual(cases[0]["function"].name, "outer")
            self.assertEqual(cases[0]["function"].source, source)
            with self.assertRaisesRegex(lint.LintError, "omit --mock"):
                core.DecisionsService(cases, core.baseline_prompt("SL001"), live=False)

    def test_frozen_grouped_data_and_review_labels(self):
        result = core.validate_dataset(DATA)
        self.assertEqual(result["counts"], {"train": 8, "dev": 8, "test": 8, "review": 4})
        self.assertTrue(all(row["label"] is None for row in core.load_split(DATA, "review")))

    def test_changed_data_and_cross_split_groups_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data"
            shutil.copytree(DATA, data)
            path = data / "test.jsonl"
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(lint.LintError):
                core.load_split(data, "test")
            shutil.copy2(DATA / "test.jsonl", path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["group"] = "direct-alias"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = core.read_json(data / "manifest.json")
            manifest["files"]["test"]["sha256"] = core.digest(path)
            (data / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(lint.LintError, "crosses splits"):
                core.validate_dataset(data)

    def test_metrics_and_probability_validation(self):
        result = core.metrics([{"probability": 0.8, "label": True},
                               {"probability": 0.9, "label": False}], 0.8)
        self.assertEqual(result["confusion"], {"tp": 1, "fp": 1, "tn": 0, "fn": 0})
        self.assertEqual((result["precision"], result["recall"]), (0.5, 1.0))
        self.assertAlmostEqual(result["brier"], 0.425)
        self.assertIsNone(core.metrics([{"probability": 0.1, "label": True}], 0.8)["precision"])
        for value in (True, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(lint.LintError):
                core.metrics([{"probability": value, "label": True}], 0.8)

    def test_mock_baseline_cli_needs_no_key_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(socket, "create_connection",
                side_effect=AssertionError("offline means no network")):
            output = Path(folder) / "baseline.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data", str(DATA), "baseline", "--mock", "--out", str(output)]), 0)
            result = core.read_json(output)
            self.assertEqual(result["mode"], "mock")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--data", str(DATA), "baseline", "--mock", "--out", str(output)]), 2)

    def test_evaluation_defaults_use_model_predictions_instead_of_fixture_scores(self):
        prompt = core.baseline_prompt("SL001")
        dev, test = core.load_split(DATA, "dev"), core.load_split(DATA, "test")
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
                patch("slop_linter.judge", return_value=(0.25, "test/model")) as judge, \
                contextlib.redirect_stdout(io.StringIO()):
            service = core.DecisionsService(dev, prompt)
            self.assertEqual(service.predict(dev[0]["function"].source, prompt["instruction"]),
                             (0.25, "test/model"))
            baseline = Path(folder) / "baseline.json"
            self.assertEqual(main(["--data", str(DATA), "baseline", "--out", str(baseline)]), 0)
            result = core.read_json(baseline)
            self.assertEqual(result["mode"], "live")
            self.assertTrue(all(row["probability"] == 0.25 for row in result["baseline"]["predictions"]))

            candidate, comparison = Path(folder) / "candidate.json", Path(folder) / "comparison.json"
            core.save_json(candidate, {"schema_version": 1, **core.metadata(DATA, "live"),
                "baseline": prompt, "candidate": {**prompt, "instruction": "Is this merely a call alias?"}})
            self.assertEqual(main(["--data", str(DATA), "compare", str(candidate),
                                   "--out", str(comparison)]), 0)
            self.assertEqual(core.read_json(comparison)["mode"], "live")
            self.assertEqual(judge.call_count, 1 + len(dev) + 2 * len(test))

    def test_explicit_request_budget_and_reusable_decision_cache(self):
        cases = core.load_split(DATA, "dev")
        prompt = core.baseline_prompt("SL001")
        service = core.DecisionsService(cases, prompt, live=False, max_calls=1)
        service.predict(cases[0]["function"].source, prompt["instruction"])
        service.predict(cases[0]["function"].source, prompt["instruction"])
        self.assertEqual(service.calls, 1)
        with self.assertRaises(lint.LintError):
            service.predict(cases[1]["function"].source, prompt["instruction"])
        for instruction in (None, "", " "):
            with self.subTest(instruction=instruction), self.assertRaises(lint.LintError):
                core.check_instruction(instruction)
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "decisions.json"
            first = core.DecisionsService(cases, prompt, live=False, cache_path=cache, cache_key="same")
            expected = first.predict(cases[0]["function"].source, prompt["instruction"])
            resumed = core.DecisionsService(cases, prompt, live=False, cache_path=cache, cache_key="same")
            self.assertEqual(resumed.predict(cases[0]["function"].source, prompt["instruction"]), expected)
            self.assertEqual(resumed.calls, 0)
            with self.assertRaisesRegex(lint.LintError, "checkpoint settings"):
                core.DecisionsService(cases, prompt, live=False, cache_path=cache, cache_key="different")

    def test_demo_cache_and_training_label_guard(self):
        train, dev = core.load_split(DATA, "train"), core.load_split(DATA, "dev")
        prompt = core.baseline_prompt("SL001")
        demos = [{"source": row["function"].source, "label": row["label"]} for row in train[:2]]
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "decisions.json"
            service = core.DecisionsService(dev, prompt, live=False, demo_cases=train,
                                           cache_path=cache, cache_key="demos")
            source = dev[0]["function"].source
            service.predict(source, prompt["instruction"])
            service.predict(source, prompt["instruction"], demos)
            service.predict(source, prompt["instruction"], list(reversed(demos)))
            service.predict(source, prompt["instruction"], demos)
            self.assertEqual(service.calls, 3)
            resumed = core.DecisionsService(dev, prompt, live=False, demo_cases=train,
                                           cache_path=cache, cache_key="demos")
            resumed.predict(source, prompt["instruction"], demos)
            self.assertEqual(resumed.calls, 0)
            for bad in ([{"source": source, "label": dev[0]["label"]}],
                        [{**demos[0], "label": not demos[0]["label"]}]):
                with self.assertRaisesRegex(lint.LintError, "reviewed training"):
                    service.predict(source, prompt["instruction"], bad)


@unittest.skipUnless(dspy is not None, "optional DSPy eval dependency not installed")
class DSPyIntegrationTests(unittest.TestCase):
    def test_mipro_searches_reviewed_grouped_demos_and_saves_the_runtime_request(self):
        train, dev = core.load_split(DATA, "train"), core.load_split(DATA, "dev")
        prompt = core.baseline_prompt("SL001")
        for case in train + dev:
            case["mock_probabilities"] = {
                "baseline": 0.01 if case["label"] else 0.99,
                "candidate": 0.99 if case["label"] else 0.01,
            }
        real_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.name in ("test.jsonl", "review.jsonl"):
                raise AssertionError("held-out examples must not enter optimization")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_open), \
                patch.object(socket, "create_connection", side_effect=AssertionError("no network")), \
                contextlib.redirect_stdout(io.StringIO()):
            artifact, service, reflection = optimize(DATA, train, dev, prompt, live=False)
        self.assertEqual(artifact["optimizer"]["name"], "dspy.MIPROv2")
        self.assertTrue(artifact["candidate"]["demos"])
        self.assertGreater(reflection.calls, 0)
        sources = {row["function"].source: row for row in train}
        for entry in artifact["optimizer"]["candidates"]:
            chosen = {demo["source"] for demo in entry["demos"]}
            for demo in entry["demos"]:
                self.assertEqual(demo["label"], sources[demo["source"]]["label"])
                self.assertTrue({source for source, row in sources.items()
                                 if row["group"] == sources[demo["source"]]["group"]} <= chosen)
        request = next(row for row in service.requests if isinstance(row["state"], dict))
        self.assertEqual(set(request["state"]["inputs"]), {"source"})
        for demo in request["state"]["demos"]:
            self.assertEqual(set(demo), {"source", "SL001"})
            self.assertIs(type(demo["SL001"]), bool)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.json"
            core.save_json(path, artifact)
            self.assertEqual(lint.load_candidate(path), artifact["candidate"])

    def test_tight_budget_preserves_a_complete_candidate_evaluation(self):
        train, dev = core.load_split(DATA, "train"), core.load_split(DATA, "dev")
        prompt = core.baseline_prompt("SL001")
        for case in train + dev:
            case["mock_probabilities"] = {
                "baseline": 0.01 if case["label"] else 0.99,
                "candidate": 0.99 if case["label"] else 0.01,
            }
        with contextlib.redirect_stdout(io.StringIO()):
            artifact, service, reflection = optimize(
                DATA, train, dev, prompt, live=False, max_calls=23, optimizer_name="gepa")
        self.assertGreater(reflection.calls, 0)
        self.assertEqual(artifact["candidate"]["instruction"], MOCK_PROPOSAL)
        for result in artifact["development"].values():
            self.assertEqual(result["metrics"]["count"], len(dev))
            self.assertEqual(len(result["predictions"]), len(dev))

    def test_classifier_failure_is_not_saved_as_a_zero_score(self):
        train, dev = core.load_split(DATA, "train"), core.load_split(DATA, "dev")
        with patch("slop_linter.judge", side_effect=lint.LintError("classifier unavailable")) as judge, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(lint.LintError, "classifier unavailable"):
                optimize(DATA, train, dev, core.baseline_prompt("SL001"),
                         live=False, max_calls=48, optimizer_name="gepa")
        self.assertEqual(judge.call_count, 1)

    def test_reflection_failure_stops_without_retry_or_result_file(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
                patch("slop_linter.judge", return_value=(0.5, "test/model")), \
                patch("evals.codex.CodexLM") as codex, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            codex.return_value.kwargs = {}
            codex.return_value.side_effect = lint.LintError("reflection unavailable")
            output = Path(folder) / "candidate.json"
            status = main(["--data", str(DATA), "optimize", "--optimizer", "gepa",
                           "--max-calls", "48", "--out", str(output)])
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())
            self.assertEqual(codex.return_value.call_count, 1)

    def test_gepa_accepts_longer_questions_and_has_no_four_revision_limit(self):
        instruction = (
            "Does this function merely rename, delegate to, or preset arguments for an existing operation, "
            "including through intermediate assignments or an elaborate docstring, while adding no "
            "substantive validation, error handling, data conversion, resource ownership, required protocol "
            "implementation, or other meaningful behavior that a direct call would lack, "
            "rather than judging it by its length or by whether it contains a function call?"
        )
        train, dev = core.load_split(DATA, "train"), core.load_split(DATA, "dev")
        labels = {case["function"].source: case["label"] for case in train + dev}
        def judge(function, api_key, model, timeout, rule):
            correct = rule.predicate == instruction
            return (0.9 if labels[function.source] == correct else 0.1, "test/model")
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
                patch("slop_linter.judge", side_effect=judge), \
                patch("evals.codex.CodexLM") as codex, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            codex.return_value.kwargs = {}
            codex.return_value.return_value = [f"```\n{instruction}\n```"]
            artifact, _, reflection = optimize(DATA, train, dev,
                core.baseline_prompt("SL001"), max_calls=96, optimizer_name="gepa")
        self.assertEqual(artifact["candidate"]["instruction"], instruction)
        self.assertGreater(reflection.calls, 4)

    def test_native_predictor_trace_and_exact_production_request(self):
        cases = core.load_split(DATA, "dev")
        prompt = core.baseline_prompt("SL001")
        service = core.DecisionsService(cases, prompt, live=False)
        program = DecisionProgram(service, MOCK_PROPOSAL)
        with dspy.context(trace=[]):
            program(source=cases[0]["function"].source)
            self.assertEqual(len(dspy.settings.trace), 1)
            self.assertIs(dspy.settings.trace[0][0], program.judge)
        request = service.requests[0]
        self.assertEqual(request["questions"]["SL001"]["instructions"], MOCK_PROPOSAL)
        self.assertEqual(request["state"], cases[0]["function"].source)
        seen = []
        def http(request, **kwargs):
            seen.append(json.loads(request.data))
            return io.BytesIO(b'{"model":"mock/decisions","answers":{"SL001":{"type":"noul","noul":0.95}}}')
        with patch("urllib.request.OpenerDirector.open", side_effect=http):
            lint.judge(cases[0]["function"], "mock-key", prompt["model"], prompt["timeout"],
                       replace(lint.RULES["SL001"], predicate=MOCK_PROPOSAL))
        self.assertEqual(seen[0], request)

    def test_real_gepa_mock_compile_has_no_test_access_and_saved_comparison(self):
        train, dev = core.load_split(DATA, "train"), core.load_split(DATA, "dev")
        prompt = core.baseline_prompt("SL001")
        original = lint.RULES["SL001"]
        real_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.name in ("test.jsonl", "review.jsonl"):
                raise AssertionError("held-out examples are not available during optimization")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_open), patch.object(socket, "create_connection",
                side_effect=AssertionError("offline means no network")), contextlib.redirect_stdout(io.StringIO()):
            artifact, service, reflection = optimize(DATA, train, dev, prompt, live=False, max_calls=48, optimizer_name="gepa")
        self.assertEqual(artifact["candidate"]["instruction"], MOCK_PROPOSAL)
        self.assertNotEqual(artifact["candidate"]["instruction"], prompt["instruction"])
        self.assertTrue(any(request["questions"]["SL001"]["instructions"] == MOCK_PROPOSAL
                            for request in service.requests))
        self.assertGreater(reflection.calls, 0)
        messages = "\n".join(message["content"] for batch in reflection.prompts for message in batch)
        for case in core.load_split(DATA, "test"):
            self.assertNotIn(case["function"].source, messages)
        self.assertIs(lint.RULES["SL001"], original)
        with tempfile.TemporaryDirectory() as folder:
            path, comparison = Path(folder) / "candidate.json", Path(folder) / "comparison.json"
            core.save_json(path, artifact)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data", str(DATA), "compare", str(path), "--mock", "--out", str(comparison)]), 0)
            result = core.read_json(comparison)
            self.assertEqual(result["split"], "test")
            self.assertEqual(result["baseline"]["metrics"]["count"], 8)
            self.assertEqual(result["mode"], "mock")
            self.assertEqual(result["candidate"]["prompt"], artifact["candidate"])
            artifact["manifest_sha256"] = "changed"
            changed = Path(folder) / "changed.json"
            core.save_json(changed, artifact)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--data", str(DATA), "compare", str(changed), "--mock", "--out", str(Path(folder) / "bad.json")]), 2)


if __name__ == "__main__":
    unittest.main()
