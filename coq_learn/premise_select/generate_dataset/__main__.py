from typing import Iterator
import pathlib
import argparse
import os
import gc
import random
from tqdm import tqdm

import coq_tracer_api
from coq_tracer_api.tracer_types import (
    Declaration,
    DeclarationKindInteractive,
    StepKindTactic,
    Event,
    EventSequence,
    EventDispatch,
    EventTactic,
    EventMessage,
    Goal,
    Hyp,
    HypKindAssumption,
    HypKindDefinition,
)

from .. import Sample


def hyp_to_string(hyp: Hyp) -> str:
    match hyp.kind:
        case HypKindAssumption():
            return f"{hyp.name} : {hyp.type_.default}"
        case HypKindDefinition(value=value):
            return f"{hyp.name} : {hyp.type_.default} = {value.default}"


def goal_to_string(goal: Goal) -> str:
    return f"{", ".join(map(hyp_to_string, goal.hyps))} ⊢ {goal.concl.default}"


def goals_to_string(goals: list[Goal]) -> str:
    return "\n".join(map(goal_to_string, goals))


def filter_goals(goals_before: list[Goal], goals_after: list[Goal]) -> list[Goal]:
    return list(set(goals_before) - set(goals_after))


def process_event(event: Event) -> Iterator[Sample]:
    match event:
        case EventSequence(elements=elements):
            for branch in elements:
                yield from process_event(branch)
        case EventDispatch(branches=branches):
            for branch in branches:
                yield from process_event(branch)
        case EventTactic(goals_before=goals_before, goals_after=goals_after, tactic=tactic):
            filtered_goals = filter_goals(goals_before, goals_after)
            if filtered_goals != []:
                for reference in tactic.references:
                    yield Sample(
                        theorem_path=declaration.path,
                        goals=goals_to_string(filtered_goals),
                        premise=reference.path,
                    )
        case EventMessage():
            pass


def process_declaration(declaration: Declaration, detailed: bool) -> Iterator[Sample]:
    match declaration.kind:
        case DeclarationKindInteractive(steps=steps):
            for step in steps:
                match step.kind:
                    case StepKindTactic(tactic=tactic, event=event):
                        if detailed:
                            yield from process_event(event)
                        else:
                            filtered_goals = filter_goals(step.goals_before, step.goals_after)
                            for reference in tactic.references:
                                yield Sample(
                                    theorem_path=declaration.path,
                                    goals=goals_to_string(filtered_goals),
                                    premise=reference.path,
                                )
                    case _:
                        pass
        case _:
            pass


gc.disable()
random.seed(0)

parser = argparse.ArgumentParser(description="Generate dataset from a trace file.")
parser.add_argument("--detailed", action="store_true")
parser.add_argument("input", type=pathlib.Path)
parser.add_argument("output", type=pathlib.Path)
args = parser.parse_args()

trace = coq_tracer_api.load_trace(args.input, deep=True, use_pydantic_parser=True, show_tqdm=True)

args.output.mkdir(parents=True, exist_ok=True)
with (
    open(os.path.join(args.output, "all.jsonl"), mode="w") as file_all,
    open(os.path.join(args.output, "train.jsonl"), mode="w") as file_train,
    open(os.path.join(args.output, "eval.jsonl"), mode="w") as file_eval,
):
    for declaration in tqdm(trace.declarations, leave=False):
        for sample in process_declaration(declaration, args.detailed):
            file_all.write(sample.model_dump_json() + "\n")
            if random.random() < 0.8:
                file_train.write(sample.model_dump_json() + "\n")
            else:
                file_eval.write(sample.model_dump_json() + "\n")

del trace
