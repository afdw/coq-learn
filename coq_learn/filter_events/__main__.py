from typing import Iterator
import sys
import os
import pathlib
import argparse

import coq_tracer_api
from coq_tracer_api.tracer_types import (
    Trace,
    Declaration,
    DeclarationKindInteractive,
    Step,
    StepKindTactic,
    Event,
    EventSequence,
    EventDispatch,
    EventTactic,
    EventMessage,
    TacticKindPrimitive,
    TacticKindBuiltin,
    TacticKindAlias,
    TacticKindML,
)

toplevel_tactic_kind_builtins = [
    "TacProgress",
    "TacAbstract",
    "TacThen",
    "TacDispatch",
    "TacExtendTac",
    "TacThens",
    "TacThens3parts",
    "TacDo",
    "TacTimeout",
    "TacTime",
    "TacTry",
    "TacRepeat",
    "TacOr",
    "TacOnce",
    "TacExactlyOnce",
    "TacThenCatch",
    "TacOrelse",
    "TacFirst",
    "TacSolve",
    "TacSelect",
    "interp_match_success",
]
atomic_tactic_kind_builtins = [
    "TacId",
    "TacFail",
    "TacIntroPattern",
    "TacApply",
    "TacElim",
    "TacCase",
    "TacMutualFix",
    "TacMutualCofix",
    "TacAssert",
    "TacGeneralize",
    "TacLetTac",
    "TacInductionDestruct",
    "TacReduce",
    "TacChange",
    "TacRewrite",
    "TacInversion",
]
value_tactic_kind_builtins = ["TacFun", "TacLetIn", "TacMatchGoal", "TacMatch", "TacCall"]
arg_tactic_kind_builtins = ["Reference", "ConstrMayEval", "TacFreshId", "TacPretype", "TacNumgoals", "ArgVar"]
# stop_tactic_kind_aliases = ["auto (nat_or_var_opt) (auto_using) (hintbases)"]


def process_event(event: Event, at_top: bool = True, in_ml: bool = False, keep_refine: bool = False) -> Iterator[EventTactic]:
    while True:
        match event:
            case EventTactic(
                kind=TacticKindBuiltin(s=s) as kind, details=EventTactic(kind=TacticKindBuiltin(s=t), tactic=tactic, details=details)
            ) if s in atomic_tactic_kind_builtins + value_tactic_kind_builtins and t in arg_tactic_kind_builtins:
                event = event.model_copy(update={"tactic": tactic, "details": details})
            case _:
                break
    match event:
        case EventSequence(elements=elements):
            for branch in elements:
                yield from process_event(branch, at_top=at_top, in_ml=False, keep_refine=keep_refine)
        case EventDispatch(branches=branches):
            for branch in branches:
                yield from process_event(branch, at_top=at_top, in_ml=False, keep_refine=keep_refine)
        case EventTactic(kind=kind, details=details):
            event_without_details = event.model_copy(update={"details": EventSequence(elements=[])})
            match kind, details:
                case TacticKindPrimitive(s="refine"), _ if keep_refine:
                    yield event_without_details
                case TacticKindPrimitive(), _:
                    pass
                case TacticKindBuiltin(s="<error>"), _:
                    pass
                case TacticKindBuiltin(s="TacThen"), _:
                    yield from process_event(details, at_top=True, in_ml=False, keep_refine=keep_refine)
                case TacticKindBuiltin(s=s), _ if s in toplevel_tactic_kind_builtins + atomic_tactic_kind_builtins:
                    yield event_without_details
                    yield from process_event(details, at_top=True, in_ml=False, keep_refine=keep_refine)
                case TacticKindBuiltin(s="TacDelay"), _:
                    yield from process_event(details, at_top=True, in_ml=False, keep_refine=keep_refine)
                case TacticKindBuiltin(s="TacCall"), EventTactic(
                    kind=TacticKindBuiltin(s="TacDelay" | "TacFun"), details=EventTactic(kind=TacticKindML()) as details
                ):
                    yield event_without_details
                    yield from process_event(details, at_top=True, in_ml=False, keep_refine=keep_refine)
                case TacticKindBuiltin(s=s), _ if s in value_tactic_kind_builtins + arg_tactic_kind_builtins and in_ml:
                    yield from process_event(details, at_top=False, in_ml=in_ml, keep_refine=keep_refine)
                case (
                    TacticKindBuiltin(s=s),
                    EventTactic(kind=TacticKindBuiltin(s=t)),
                ) if s in value_tactic_kind_builtins + arg_tactic_kind_builtins and t in value_tactic_kind_builtins + arg_tactic_kind_builtins + [
                    "TacDelay"
                ] and not at_top:
                    yield from process_event(details, at_top=False, in_ml=in_ml, keep_refine=keep_refine)
                case TacticKindBuiltin(s=s), _ if s in value_tactic_kind_builtins + arg_tactic_kind_builtins:
                    yield event_without_details
                    yield from process_event(details, at_top=False, in_ml=in_ml, keep_refine=keep_refine)
                case TacticKindBuiltin(s=s), _:
                    raise NotImplementedError(f"unhandled builtin: {s}")
                case TacticKindAlias(), EventTactic(kind=TacticKindBuiltin(s="TacDelay" | "TacFun"), details=EventTactic(kind=TacticKindML()) as details):
                    yield event_without_details
                    yield from process_event(details, at_top=True, in_ml=False, keep_refine=keep_refine)
                # case TacticKindAlias(s=s), _ if s in stop_tactic_kind_aliases:
                #     yield event_without_details
                case TacticKindAlias(), _:
                    yield from process_event(details, at_top=True, in_ml=False, keep_refine=keep_refine)
                case TacticKindML(s="coq-core.plugins.ltac::assumption@0"), _:
                    yield from process_event(details, at_top=True, in_ml=True, keep_refine=True)
                case TacticKindML(), _:
                    yield from process_event(details, at_top=True, in_ml=True, keep_refine=keep_refine)
        case EventMessage():
            pass


def process_step(step: Step) -> Step:
    match step.kind:
        case StepKindTactic(event=event):
            return step.model_copy(update={"kind": step.kind.model_copy(update={"event": EventDispatch(goals_before=[], branches=list(process_event(event)))})})
        case _:
            return step


def process_declaration(declaration: Declaration) -> Declaration:
    match declaration.kind:
        case DeclarationKindInteractive(steps=steps):
            return declaration.model_copy(update={"kind": declaration.kind.model_copy(update={"steps": list(map(process_step, steps))})})
        case _:
            return declaration


def process_trace(trace: Trace) -> Trace:
    return trace.model_copy(update={"declarations": map(process_declaration, trace.declarations)})


def process_file(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    print(f"{input_path} -> {output_path}", file=sys.stderr)
    if input_path.is_dir():
        output_path.mkdir(exist_ok=True)
        for file_path in os.listdir(input_path):
            process_file(input_path.joinpath(file_path), output_path.joinpath(file_path))
    else:
        trace = coq_tracer_api.load_trace(input_path)
        trace = process_trace(trace)
        coq_tracer_api.save_trace(output_path, trace)


sys.setrecursionlimit(1_000_000)

parser = argparse.ArgumentParser(description="Process a trace to filter out uninteresting events.")
parser.add_argument("input", type=pathlib.Path)
parser.add_argument("output", type=pathlib.Path)
args = parser.parse_args()

process_file(args.input, args.output)
