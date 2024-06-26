from typing import Annotated
from functools import cache
import pathlib
import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import Body, FastAPI
import uvicorn

parser = argparse.ArgumentParser(description="Run the server for inference.")
parser.add_argument("--model-name", help="Name of the model or directory containing the model", type=pathlib.Path, required=True)
parser.add_argument("-a", "--addr", type=str, default="0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=8000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

model.to(device)


@cache
def infer(s: str, num_return_sequences: int) -> list[tuple[str, float]]:
    inputs = tokenizer(s, return_tensors="pt")
    inputs = inputs.to(device)
    torch.manual_seed(0)
    outputs = model.generate(
        **inputs,
        return_dict_in_generate=True,
        output_scores=True,
        num_return_sequences=num_return_sequences,
        # num_beams=num_return_sequences,
        max_new_tokens=25,
        do_sample=True,
        temperature=2.0,
    )
    decoded_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    # print(f"{outputs.sequences_scores=}")
    # print(f"{dir(outputs)}")
    # print(f"{transition_scores=}")
    # print(f"{outputs.scores=}")
    # print(f"{transition_scores.shape=}")
    # print(f"{len(outputs.scores)=}", f"{outputs.scores[0].shape=}")
    results = [
        (
            output,
            math.exp(sum(x for x in transition_score if x != -float("inf")))
        )
            for output, transition_score
            in zip(decoded_outputs, transition_scores)
    ]
    print(f"{results=}")
    results = list(set(results))
    results.sort(key=lambda t: t[1], reverse=True)
    return results


app = FastAPI()


@app.post("/infer")
async def post_infer(s: Annotated[str, Body()], num_return_sequences: int = 10) -> list[tuple[str, float]]:
    return infer(s=s, num_return_sequences=num_return_sequences)


uvicorn.run(app, host=args.addr, port=args.port, ws_ping_timeout=None)
