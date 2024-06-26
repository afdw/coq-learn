import pathlib
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import functools
from starlette.websockets import WebSocket
from fastapi import FastAPI
import uvicorn

import coq_interact_api
from coq_interact_api import Internal, TypeDescUnit, TypeDescList, TypeDescGoal, Tactic, HypKindAssumption, HypKindDefinition, Hyp, Goal, Handler

parser = argparse.ArgumentParser(description="Run the server for proof search.")
parser.add_argument("--model-name", help="Name of the model or directory containing the model", type=pathlib.Path, required=True)
parser.add_argument("-a", "--addr", type=str, default="0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=8000)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)


@functools.cache
def infer(s: str, num_return_sequences: int, num_beams: int) -> list[str]:
    inputs = tokenizer(s, return_tensors="pt")
    # outputs = model.generate(**inputs, num_return_sequences=count, do_sample=True) # num_return_sequences <= num_beams /\ do_sample = True;;; num_beams=count * 10
    # outputs = model.generate(**inputs, num_return_sequences=num_return_sequences, num_beams=num_beams)
    torch.manual_seed(0)
    outputs = model.generate(**inputs, num_return_sequences=num_return_sequences, do_sample=True, temperature=2.0)
    return list(set(tokenizer.batch_decode(outputs, skip_special_tokens=True)))


app = FastAPI()


@app.websocket("/interact_tactic_predict")
async def websocket_interact_tactic_predict(websocket: WebSocket, depth_limit: int = 7, apply: bool = True) -> None:
    await websocket.accept()

    async def get_tactic(handler: Handler) -> Internal[Tactic[None]]:
        class DepthException(Exception):
            pass

        async def main(current_depth: int, depth_limit: int) -> Internal[Tactic[None]]:
            async def handle_goal(goal: Goal) -> Internal[Tactic[None]]:
                if depth_limit < 0:
                    return await handler.tactic_fail(DepthException())

                async def hyp_print(hyp: Hyp) -> str:
                    match hyp.kind:
                        case HypKindAssumption():
                            return f"{hyp.name} : {await handler.constr_print(hyp.type_)}"
                        case HypKindDefinition(value=value):
                            return f"{hyp.name} : {await handler.constr_print(hyp.type_)} = {await handler.constr_print(value)}"

                async def goal_print(goal: Goal) -> str:
                    return f"{", ".join([await hyp_print(hyp) for hyp in goal.hyps])} ⊢ {await handler.constr_print(goal.concl)}"

                async def apply_ltac(t: str) -> Internal[Tactic[None]]:
                    async def k(_: None) -> Internal[Tactic[None]]:
                        if apply:
                            return await handler.tactic_ltac(t)
                        else:
                            return await handler.tactic_return(None)

                    return await handler.tactic_bind(TypeDescUnit(), await handler.tactic_message(f"{"  " * current_depth}Applying: {t}"), k)

                async def k(_: None) -> Internal[Tactic[None]]:
                    return await handler.tactic_or_list([await apply_ltac(t) for t in infer(printed_goal, num_return_sequences=20, num_beams=100)])

                printed_goal = await goal_print(goal)
                return await handler.tactic_bind(
                    TypeDescUnit(),
                    await handler.tactic_message(f"{"  " * current_depth}Got goal: {printed_goal}"),
                    k,
                )

            async def k(_: list[None]) -> Internal[Tactic[None]]:
                async def k(goals: list[Goal]) -> Internal[Tactic[None]]:
                    if goals:
                        return await main(current_depth=current_depth + 1, depth_limit=depth_limit - 1)
                    else:
                        return await handler.tactic_return(None)

                return await handler.tactic_bind(TypeDescList(element_type_desc=TypeDescGoal()), await handler.tactic_goals(), k)

            return await handler.tactic_bind(
                TypeDescList(element_type_desc=TypeDescUnit()),
                await handler.tactic_enter(handle_goal),
                k,
            )

        return await main(current_depth=0, depth_limit=depth_limit)

    await coq_interact_api.handle_websocket(websocket, get_tactic)


uvicorn.run(app, host=args.addr, port=args.port, ws_ping_timeout=None)
