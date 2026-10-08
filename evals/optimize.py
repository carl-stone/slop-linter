from __future__ import annotations

from pathlib import Path

import dspy
from dspy.experimental import Noul

from .core import DecisionsService, check_instruction, digest, evaluate, metadata, read_json, save_json
from slop_linter.transport import DEMO_CONTEXT
import slop_linter as lint


class DecisionsClient:
    supports_decision_requests = True
    kwargs = {}

    def __init__(self, service: DecisionsService):
        self.service = service

    def __deepcopy__(self, memo):
        # GEPA copies programs; their request budget and cache must stay shared.
        return self

    def __call__(self, *, state, questions):
        if set(questions) != {"decision"} or questions["decision"]["type"] != "noul":
            raise lint.LintError("DSPy must request exactly one Noul output")
        if set(state["inputs"]) != {"source"}:
            raise lint.LintError("only function source may be a classifier input")
        # Put the optimized signature instruction in the actual Noul question,
        # using exactly the same source/request path as the production linter.
        demos = [{"source": demo["source"], "label": demo["decision"]}
                 for demo in state.get("demos", [])]
        p, _ = self.service.predict(state["inputs"]["source"], state["instructions"], demos)
        return {"decision": {"type": "noul", "noul": p}}


class DecisionSignature(dspy.Signature):
    source: str = dspy.InputField(desc="Python function source.")
    decision: Noul = dspy.OutputField(desc="Whether the lint predicate is true.")


class DecisionProgram(dspy.Module):
    def __init__(self, service: DecisionsService, instruction: str, demos=()):
        super().__init__()
        self.judge = dspy.Predict(DecisionSignature.with_instructions(instruction))
        self.judge.lm = DecisionsClient(service)
        self.judge.demos = [dspy.Example(source=demo["source"], decision=demo["label"])
                            for demo in demos]

    def forward(self, source):
        prediction = self.judge(source=source)
        return dspy.Prediction(probability=prediction.decision.probability)


REFLECTION_GUIDANCE = (
    "The instruction you are optimizing is for Jev, a decision model, not a chat assistant. "
    "Jev judges only the target Python function source, optionally alongside labeled example functions, "
    "and returns the probability that the predicate is true. "
    "Yes means flag a needless helper; No means keep acceptable behavior. "
    "Prefer a concise, natural decision question. Do not ask Jev for reasoning, explanations, or text output. "
    "Use the labels and rationales to learn the distinction; do not copy example-specific names into the question. "
    "The objective is to minimize mean squared error between P(Yes) and the Boolean label, "
    "not to optimize a classification cutoff. Follow the requested response format. "
    "When proposing an instruction, output a decision question; examples are supplied separately. "
    "Consider concise general questions whose distinction is communicated by the labeled examples, "
    "as well as questions that state the distinction themselves."
)


MOCK_PROPOSAL = (
    "Is this helper only a call alias or fixed-argument wrapper, "
    "without validation, data transformation, resource management, or interface adaptation?"
)


class ReflectionLM(dspy.BaseLM):
    def __init__(self, *, live=True, model, api_key, backend="codex", structured=False):
        if backend not in {"codex", "openrouter"}:
            raise lint.LintError("unknown reflection backend")
        if live and backend == "openrouter" and not model:
            raise lint.LintError("OpenRouter reflection needs --reflection-model")
        self.backend = backend if live else "mock"
        identifier = f"{backend}/{model or 'default'}" if live else "mock/reflection"
        super().__init__(model=identifier)
        self.live, self.structured = live, structured
        # MIPRO copies its prompt LM; counters and failure state stay shared.
        self.stats = {"calls": 0, "error": None}
        self.prompts = []
        if live and backend == "codex":
            from .codex import CodexLM
            self.delegate = CodexLM(model=model)
        else:
            self.delegate = dspy.LM(
                "openrouter/" + model, api_key=api_key, temperature=0.7,
                num_retries=0, cache=False,
            ) if live else None
        self.kwargs = self.delegate.kwargs.copy() if live else {}

    @property
    def calls(self):
        return self.stats["calls"]

    @property
    def error(self):
        return self.stats["error"]

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        if self.error is not None:
            raise self.error
        self.stats["calls"] += 1
        # Preserve GEPA's standard reflection prompt and add Jev-specific guidance.
        conversation = [{"role": "system", "content": REFLECTION_GUIDANCE},
                        *(messages if messages is not None else [{"role": "user", "content": prompt}])]
        if self.live:
            try:
                return self.delegate(messages=conversation, **kwargs)
            except lint.LintError as exc:
                self.stats["error"] = exc
                raise
            except Exception as exc:
                self.stats["error"] = lint.LintError(f"{self.backend} reflection failed ({type(exc).__name__})")
                raise self.error from exc
        self.prompts.append(conversation)
        if self.structured:
            system = "\n".join(message["content"] for message in conversation if message["role"] == "system")
            field = next(name for name in ("proposed_instruction", "summary", "observations")
                         if f"[[ ## {name} ## ]]" in system)
            content = MOCK_PROPOSAL if field == "proposed_instruction" else (
                "True labels identify aliases and argument presets; False labels identify "
                "validation, transformation, and required protocols.")
            return [f"[[ ## {field} ## ]]\n{content}\n[[ ## completed ## ]]"]
        return [f"```\n{MOCK_PROPOSAL}\n```"]


class LabeledMIPRO(dspy.MIPROv2):
    proposal_path = None

    def _bootstrap_fewshot_examples(self, program, trainset, seed, teacher, *,
                                   num_fewshot_candidates, **kwargs):
        # DSPy 3.4's stock generator calls randint(1, 0) with bootstrapping
        # disabled. Use LabeledFewShot to build gold-label sets instead. Keep
        # paired variants together and let the search vary the number of demos
        # over the available training groups, including the original program.
        groups = {}
        for example in trainset:
            groups.setdefault(example.group, []).append(example)
        pool = list(groups.values())
        pairs = sorted((group for group in pool if {row.label for row in group} == {True, False}), key=len)
        sets = {i: [list(pred.demos)] for i, pred in enumerate(program.predictors())}
        for index in range(1, num_fewshot_candidates):
            selected = (pairs[index - 1] if index <= len(pairs) else
                        [row for group in self.rng.sample(pool, self.rng.randint(1, len(pool)))
                         for row in group])
            sampled = dspy.LabeledFewShot(k=len(selected)).compile(program, trainset=selected, sample=False)
            for i, pred in enumerate(sampled.predictors()):
                sets[i].append(pred.demos)
        self.labeled_sets = sets
        return sets

    def _propose_instructions(self, program, trainset, demo_candidates, *args, **kwargs):
        count = kwargs["num_instruct_candidates"]
        if self.proposal_path is not None and self.proposal_path.exists():
            saved = read_json(self.proposal_path)
            if saved.get("count") != count:
                raise lint.LintError("proposal pool size changed; choose a new log-dir")
            instructions = {int(i): [check_instruction(text) for text in values]
                            for i, values in saved["instructions"].items()}
        else:
            # The stock proposer displays only demos tagged as augmented.
            # Tag copies for its formatter so it can see our native gold labels;
            # the actual inference demos remain unchanged.
            proposal_demos = {i: [
                [dspy.Example(**dict(demo.items()), augmented=True) for demo in group]
                for group in groups] for i, groups in demo_candidates.items()}
            instructions = super()._propose_instructions(program, trainset, proposal_demos, *args, **kwargs)
            if self.proposal_path is not None:
                save_json(self.proposal_path, {"count": count, "instructions": instructions})
        return instructions


def optimize(data, train, dev, prompt, *, live=True, reflection_model=None,
             reflection_backend="codex", max_calls=None, auto=None, seed=7, log_dir=None,
             optimizer_name="mipro", num_trials=None, num_candidates=None):
    if dspy.__version__ != "3.4.0":
        raise lint.LintError("install the pinned eval extra: DSPy 3.4.0")
    if max_calls is not None and (type(max_calls) is not int or max_calls <= 0):
        raise lint.LintError("max-calls must be a positive integer")
    if auto is not None and max_calls is not None:
        raise lint.LintError("choose either a GEPA auto budget or max-calls")
    log_dir = Path(log_dir) if log_dir is not None else None
    if optimizer_name not in {"mipro", "gepa"}:
        raise lint.LintError("unknown optimizer")
    if optimizer_name == "mipro" and max_calls is not None:
        raise lint.LintError("MIPRO uses --auto or --trials with --num-candidates; --max-calls is for GEPA")
    cache_key = {"manifest": digest(data / "manifest.json"), "prompt": prompt,
                 "mode": "live" if live else "mock", "seed": seed,
                 "reflection_backend": reflection_backend, "reflection_model": reflection_model,
                 "reflection_guidance": REFLECTION_GUIDANCE, "optimizer": optimizer_name,
                 "demo_context": DEMO_CONTEXT}
    service = DecisionsService(train + dev, prompt, live=live,
        cache_path=log_dir / "decisions.json" if log_dir else None, cache_key=cache_key, demo_cases=train)
    reflection = ReflectionLM(live=live, model=reflection_model,
                              api_key=service.api_key, backend=reflection_backend,
                              structured=optimizer_name == "mipro")
    student = DecisionProgram(service, prompt["instruction"], prompt.get("demos", []))

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        p = pred.probability
        loss = (p - int(gold.label)) ** 2
        return dspy.Prediction(score=1 - loss,
            feedback=f"Expected {'Yes (flag)' if gold.label else 'No (keep)'}; "
                     f"P(Yes)={p:.3f}; squared error={loss:.4f}. {gold.rationale}")

    def examples(cases):
        return [dspy.Example(source=case["function"].source, label=case["label"],
                             decision=case["label"], group=case["group"],
                             rationale=case["rationale"]).with_inputs("source") for case in cases]

    def stop(state):
        if service.error is not None:
            raise service.error
        if reflection.error is not None:
            raise reflection.error
        return False

    if optimizer_name == "mipro":
        if (num_trials is None) != (num_candidates is None):
            raise lint.LintError("provide both --trials and --num-candidates, or use --auto")
        if auto is not None and num_trials is not None:
            raise lint.LintError("choose --auto or explicit MIPRO trials and candidates")
        if any(type(value) is not int or value <= 0 for value in (num_trials, num_candidates) if value is not None):
            raise lint.LintError("trials and num-candidates must be positive integers")
        optimizer = LabeledMIPRO(
            metric=lambda gold, pred, trace=None: metric(gold, pred).score,
            prompt_model=reflection, task_model=student.judge.lm,
            max_bootstrapped_demos=0, max_labeled_demos=len(train),
            auto=(auto or "medium") if num_trials is None else None,
            num_candidates=num_candidates, num_threads=1, max_errors=1, seed=seed,
        )
        optimizer.proposal_path = log_dir / "instructions.json" if log_dir else None
    else:
        if num_trials is not None or num_candidates is not None:
            raise lint.LintError("trials and num-candidates are for MIPRO")
        optimizer = dspy.GEPA(
            metric=metric, max_metric_calls=max_calls,
            auto=(auto or "medium") if max_calls is None else None, reflection_lm=reflection,
            use_merge=False, num_threads=1, track_stats=True, seed=seed,
            log_dir=str(log_dir) if log_dir else None,
            gepa_kwargs={"stop_callbacks": stop},
        )
    try:
        options = ({"num_trials": num_trials, "minibatch": False,
                    "program_aware_proposer": False, "tip_aware_proposer": False,
                    "fewshot_aware_proposer": True, "view_data_batch_size": len(train)}
                   if optimizer_name == "mipro" else {})
        compiled = optimizer.compile(student, trainset=examples(train), valset=examples(dev), **options)
    except Exception as exc:
        error = service.error or reflection.error
        if error is not None:
            if error is exc:
                raise
            raise error from exc
        if isinstance(exc, lint.LintError):
            raise
        raise lint.LintError(f"DSPy optimization failed: {exc}") from exc
    # GEPA isolates worker/proposer errors; those must never become classifier scores.
    if service.error is not None:
        raise service.error
    if reflection.error is not None:
        raise reflection.error
    instruction = check_instruction(compiled.judge.signature.instructions)
    candidate = {**prompt, "instruction": instruction}
    if compiled.judge.demos:
        candidate["demos"] = [{"source": demo.source, "label": demo.decision}
                              for demo in compiled.judge.demos]
    else:
        candidate.pop("demos", None)
    if optimizer_name == "gepa":
        details = compiled.detailed_results
        search = {"name": "dspy.GEPA", "max_metric_calls": optimizer.max_metric_calls,
                  "total_metric_calls": details.total_metric_calls,
                  "selected_candidate": details.best_idx,
                  "candidate_instructions": [program.judge.signature.instructions for program in details.candidates],
                  "development_scores": details.val_aggregate_scores}
    else:
        search = {"name": "dspy.MIPROv2", "trials": len(compiled.candidate_programs) - 1,
                  "demonstration_source": "reviewed training labels; related groups kept together",
                  "demonstration_candidates": len(optimizer.labeled_sets[0]),
                  "instruction_pool": str(optimizer.proposal_path) if optimizer.proposal_path else None,
                  "candidates": [{"instruction": entry["program"].judge.signature.instructions,
                      "demos": [{"source": demo.source, "label": demo.decision}
                                for demo in entry["program"].judge.demos],
                      "development_score": entry["score"] / 100}
                      for entry in compiled.candidate_programs]}
    result = {"schema_version": 1, **metadata(data, "live" if live else "mock"),
        "baseline": prompt, "candidate": candidate,
        "optimizer": {**search, "dspy_version": dspy.__version__, "seed": seed,
            "metric": "1 - per-example Brier loss", "auto": optimizer.auto,
            "decision_calls": service.calls,
            "reflection_calls": reflection.calls, "reflection_backend": reflection.backend,
            "reflection_model": reflection.model,
            "log_dir": str(log_dir) if log_dir else None,
            "training_examples_available": len(train),
            "training_examples_scored": sum(any(source == case["function"].source
                for source, instruction, demos in service.cache) for case in train),
            "resolved_decision_models": sorted(service.models)},
        "development": {"baseline": evaluate(dev, prompt, service),
                        "candidate": evaluate(dev, candidate, service)}}
    return result, service, reflection
