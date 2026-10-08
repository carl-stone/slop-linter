from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
from unittest.mock import patch

import slop_linter as lint
from slop_linter import check_instruction, validate_prompt
from slop_linter.transport import DEMO_CONTEXT


DATA_ROOT = Path(__file__).parent / "data"


def read_json(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise lint.LintError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise lint.LintError(f"JSON must contain an object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def manifest(data: Path) -> dict:
    value = read_json(data / "manifest.json")
    if value.get("schema_version") != 1 or value.get("rule") not in lint.RULES:
        raise lint.LintError("unsupported dataset schema or rule")
    return value


def load_split(data: Path, split: str) -> list[dict]:
    metadata = manifest(data)
    path = data / f"{split}.jsonl"
    if digest(path) != metadata["files"][split]["sha256"]:
        raise lint.LintError(f"{split} data changed; create a new dataset version")
    records = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise lint.LintError(f"invalid JSONL in {split}") from exc
        if not isinstance(row, dict):
            raise lint.LintError(f"examples in {split} must be JSON objects")
        if (row.get("split") != split
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in ("id", "group", "source", "rationale"))):
            raise lint.LintError(f"invalid example metadata in {split}")
        if (split == "review" and row.get("label") is not None
                or split != "review" and type(row.get("label")) is not bool):
            raise lint.LintError(f"invalid label in {split}")
        functions = lint.extract_functions(row["id"] + ".py", row["source"].encode())
        if not functions:
            raise lint.LintError("each example must contain a function")
        # The first definition is the target; nested definitions are its source context.
        row["function"] = functions[0]
        records.append(row)
    if len(records) != metadata["files"][split]["count"]:
        raise lint.LintError(f"incorrect example count for {split}")
    return records


def validate_dataset(data: Path) -> dict:
    metadata = manifest(data)
    ids, sources, groups, counts, labels = set(), set(), {}, {}, {}
    for split in ("train", "dev", "test", "review"):
        cases = load_split(data, split)
        counts[split] = len(cases)
        labels[split] = {"flag": sum(case["label"] is True for case in cases),
                         "keep": sum(case["label"] is False for case in cases)}
        if split != "review" and {case["label"] for case in cases} != {True, False}:
            raise lint.LintError(f"{split} needs positive and negative examples")
        for case in cases:
            if case["id"] in ids or case["source"] in sources:
                raise lint.LintError("duplicate example ID or source")
            ids.add(case["id"])
            sources.add(case["source"])
            if groups.setdefault(case["group"], split) != split:
                raise lint.LintError("related example group crosses splits")
    return {"version": metadata["version"], "counts": counts, "labels": labels,
            "groups": len(groups)}


def baseline_prompt(rule: str, model: str | None = None, threshold: float = lint.DEFAULT_THRESHOLD) -> dict:
    return {"rule": rule, "instruction": lint.RULES[rule].predicate,
            "model": model or lint.DEFAULT_MODEL, "threshold": threshold,
            "timeout": 30.0}


class DecisionsService:
    def __init__(self, cases: list[dict], prompt: dict, *, live=True, max_calls=None,
                 cache_path: Path | None = None, cache_key=None, demo_cases=None):
        validate_prompt(prompt)
        self.prompt = prompt
        self.live = live
        self.max_calls = max_calls
        self.calls = 0
        self.error = None
        self.requests = []
        self.models = set()
        self.cache = {}
        self.cache_path, self.cache_key = cache_path, cache_key
        self.lock = threading.RLock()
        self.cases = {case["function"].source: case for case in cases}
        self.demo_labels = ({case["function"].source: case["label"] for case in demo_cases}
                            if demo_cases is not None else None)
        if len(self.cases) != len(cases):
            raise lint.LintError("duplicate extracted function source")
        if not live and any("mock_probabilities" not in case for case in cases):
            raise lint.LintError("this dataset has no mock probabilities; omit --mock to evaluate Jev")
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() if live else "offline-mock-key"
        if live and (not self.api_key or any(char.isspace() for char in self.api_key)):
            raise lint.LintError("live calls need OPENROUTER_API_KEY")
        if cache_path is not None and cache_path.exists():
            saved = read_json(cache_path)
            if saved.get("key") != cache_key:
                raise lint.LintError("checkpoint settings or dataset changed; choose a new log-dir")
            for row in saved["predictions"]:
                if row["source"] not in self.cases or not 0 <= row["probability"] <= 1:
                    raise lint.LintError("invalid cached decision")
                demos = json.dumps(row.get("demos", []), sort_keys=True)
                self.cache[(row["source"], row["instruction"], demos)] = (row["probability"], row["model"])
                self.models.add(row["model"])

    def predict(self, source: str, instruction: str, demos=None) -> tuple[float, str]:
        check_instruction(instruction)
        demos = [] if demos is None else demos
        validate_prompt({**self.prompt, "instruction": instruction, "demos": demos})
        if self.demo_labels is not None and any(
            demo["source"] not in self.demo_labels or self.demo_labels[demo["source"]] != demo["label"]
            for demo in demos
        ):
            raise lint.LintError("demonstrations must use reviewed training labels")
        key = (source, instruction, json.dumps(demos, sort_keys=True))
        with self.lock:
            if self.error is not None:
                raise self.error
            if key in self.cache:
                return self.cache[key]
            if source not in self.cases:
                raise lint.LintError("classifier received source outside the allowed split")
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise lint.LintError("decision request budget exhausted")
            self.calls += 1
            case = self.cases[source]
            rule = replace(lint.RULES[self.prompt["rule"]], predicate=instruction, demos=tuple(demos))
            try:
                if self.live:
                    result = lint.judge(case["function"], self.api_key, self.prompt["model"],
                                        self.prompt["timeout"], rule)
                else:
                    kind = "baseline" if instruction == lint.RULES[rule.code].predicate and not demos else "candidate"
                    probability = case["mock_probabilities"][kind]
                    def mocked_http(request, **kwargs):
                        payload = json.loads(request.data)
                        state = ({"instructions": DEMO_CONTEXT,
                                  "input_fields": "source: Python function source to assess.",
                                  "inputs": {"source": source}, "demos": [
                            {"source": demo["source"], rule.code: demo["label"]} for demo in demos]}
                            if demos else source)
                        assert payload == {"model": self.prompt["model"], "state": state,
                                           "questions": {rule.code: {"type": "noul", "instructions": instruction}}}
                        self.requests.append(payload)
                        return io.BytesIO(json.dumps({"model": "mock/decisions",
                            "answers": {rule.code: {"type": "noul", "noul": probability}}}).encode())
                    with patch("urllib.request.OpenerDirector.open", side_effect=mocked_http):
                        result = lint.judge(case["function"], self.api_key, self.prompt["model"],
                                            self.prompt["timeout"], rule)
            except lint.LintError as exc:
                self.error = exc
                raise
            self.models.add(result[1])
            self.cache[key] = result
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"key": self.cache_key, "predictions": [
                    {"source": source, "instruction": instruction, "demos": json.loads(demos),
                     "probability": p, "model": model}
                    for (source, instruction, demos), (p, model) in self.cache.items()
                ]}, indent=2, allow_nan=False) + "\n")
                temporary.replace(self.cache_path)
            return result


def metrics(rows: list[dict], threshold: float) -> dict:
    if not rows:
        raise lint.LintError("cannot evaluate an empty split")
    tp = fp = tn = fn = 0
    brier = log_loss = 0.0
    for row in rows:
        p, label = row["probability"], row["label"]
        if type(p) not in (int, float) or not 0 <= p <= 1:
            raise lint.LintError("invalid evaluation probability")
        predicted = p >= threshold
        tp += int(predicted and label)
        fp += int(predicted and not label)
        tn += int(not predicted and not label)
        fn += int(not predicted and label)
        brier += (p - int(label)) ** 2
        q = min(1 - 1e-12, max(1e-12, p))
        log_loss -= math.log(q if label else 1 - q)
    return {"count": len(rows), "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "accuracy": (tp + tn) / len(rows), "brier": brier / len(rows),
            "log_loss": log_loss / len(rows)}


def evaluate(cases: list[dict], prompt: dict, service: DecisionsService) -> dict:
    rows = []
    for case in cases:
        p, model = service.predict(case["function"].source, prompt["instruction"], prompt.get("demos", []))
        rows.append({"id": case["id"], "group": case["group"], "label": case["label"],
                     "probability": p, "predicted": p >= prompt["threshold"], "model": model})
    return {"prompt": prompt, "metrics": metrics(rows, prompt["threshold"]), "predictions": rows}


def metadata(data: Path, mode: str) -> dict:
    value = manifest(data)
    return {"created_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "data_version": value["version"], "manifest_sha256": digest(data / "manifest.json")}
