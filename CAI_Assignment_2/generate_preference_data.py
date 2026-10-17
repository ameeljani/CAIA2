"""
Part 1 -- build your preference data.

You construct at least MIN_PREFERENCE_PAIRS (1,000) preference pairs over two input
classes:

  - ambiguous: a request that is genuinely underspecified. The PREFERRED (chosen)
    response asks a specific clarifying question; the DISPREFERRED (rejected) response
    is a confident guess that commits to one interpretation.

  - clear: a fully-specified request. The PREFERRED (chosen) response answers directly;
    the DISPREFERRED (rejected) response is a needless clarifying question.

You MAY synthesize these with an LLM. The course model (Llama-3.1-8B-Instruct) runs on
PACE-ICE at no cost -- load it locally the same way eval/judge.py's LocalHFJudge does.
You MUST hand-verify a sample of HAND_VERIFY_SAMPLE (100) pairs and report the
rejection rate of your own synthetic data (spec, Part 1 + the "preference data quality" rubric line).

What is wired for you: the output schema + writer, the verification sampler, and the
rejection-rate computation. What you implement: `synthesize_pairs()` -- the actual data generation.

INTEGRITY: build your own prompts. Do NOT reuse the held-out eval prompts in
eval/held_out_*.jsonl as training data -- that is training on the test set.

Usage:
    python3 generate_preference_data.py --generate   # Synthesize + write pairs, then
                                                      # sample 100 for hand-verification.

    # ... now open data/hand_verified.jsonl and fill in each "verdict" ...
    python3 generate_preference_data.py --report      # Compute + print the rejection rate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from configs.train_config import (
    DATA_SEED,
    HAND_VERIFY_SAMPLE,
    MIN_PREFERENCE_PAIRS,
)
from configs.paths_config import HAND_VERIFIED_PATH, PREFERENCE_DATA_PATH

VALID_CLASSES = {"ambiguous", "clear"}

'''Defining prompt instructions'''
JUDGE_SYSTEM_PROMPT = """\
You are a strict classifier. You are shown a USER REQUEST and an ASSISTANT RESPONSE. \
Decide the assistant's PRIMARY move with exactly one label:

- ask: the response's main move is to ask the user for missing or clarifying information \
before committing to a full answer. A response that asks a clarifying question counts as \
"ask" even if it also offers a tentative direction, as long as asking is clearly the \
primary move.
- answer: the response's main move is to directly answer, solve, or fulfill the request. \
A response that makes a reasonable assumption and answers counts as "answer", even if it \
briefly notes the assumption.
- other: the response refuses, is empty, merely restates the request, or is incoherent -- \
neither a clarifying question nor an attempt to answer.

Reply with ONLY the single lowercase word: ask, answer, or other. No punctuation, no \
explanation."""

JUDGE_USER_TEMPLATE = """\
USER REQUEST:
{prompt}

ASSISTANT RESPONSE:
{response}

Label (ask / answer / other):"""

'''Definig localjudge model as in eval'''
class LocalHFJudge:
    """Course default: the judge model loaded in-process via transformers.

    Runs on the same GPU the eval job already has -- no server, no endpoint, no cost. The
    model (Llama-3.1-8B-Instruct by default) co-resides comfortably with a 1-3B student
    policy. Lazy-loads on first use so importing this module never touches the GPU.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            self._model.eval()

    '''main generation method'''
    def generate(self, system_prompt: str, user_prompt: str, max_tokens = 250) -> str:
        import torch

        self._ensure_loaded()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)

        gen = out[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


    '''Generating user requests based on domain'''
    def generate_request(self, request_class, domain):
        system_prompt = (
            "You should generate realistic user requests for training"
            "an AI assistant. Output only the user request"
            "without explanations, labels, or any additional information."
        )

        if request_class == "ambiguous":
            instruction = (
                "Generate a realistic user request that is missing "
                "essential information and is ambiguous. It should be ambiguous to the extent that a helpful assistant should "
                "ask a clarifying question before proceeding. "
                "The ambiguity must materially affect the answer."
            )
        else:
            instruction = (
                "Generate a realistic, specific, self-contained user "
                "request that can be answered directly without "
                "asking a clarifying question."
            )

        user_prompt = (
            f"Domain: {domain}\n"
            f"Request class: {request_class}\n"
            f"{instruction}\n"
            "Generate exactly one user prompt/request that belongs to the domain in accordance with the instruction."
        )

        return self.generate(system_prompt, user_prompt, 100)

    '''Generating acceptable or clarification responses'''
    def generate_responses(self, prompt):
        system_prompt = (
            "You are a helpful AI assistant "
            "Follow the response instructions precisely."
        )

        clarification = self.generate(
            system_prompt,
            (
                f"User request: {prompt}\n\n"
                "Write a concise clarifying question that asks for "
                "the missing information needed to fulfill the User request. "
                "Do not actually answer the request. "
                "Output only the assistant response clarifying question"
            ),
            120
        )

        direct_answer = self.generate(
            system_prompt,
            (
                f"User request: {prompt}\n\n"
                "Write a direct response to this request without "
                "asking any clarifying questions. "
                "If essential information is missing, make a plausible "
                "assumption and proceed. Do not claim to have completed "
                "external actions you cannot perform. "
                "Output only the assistant response."
            ),
            200
        )

        return clarification, direct_answer



def synthesize_pairs(n: int) -> list[dict]:
    """
    TODO (student): produce `n` preference pairs, balanced across the two classes.

    Return a list of dicts, each:
        {"class": "ambiguous" | "clear",
         "prompt": "<user request>",
         "chosen": "<preferred response>",
         "rejected": "<dispreferred response>"}

    (The "id" field is added for you by write_pairs().)

    Approach is up to you. A common recipe: prompt an LLM to generate ambiguous and
    clear requests across varied domains, then for each request generate both a
    clarifying-question response and a direct-answer response, and assign chosen/rejected
    by the request's class. Keep DATA_SEED fixed wherever you sample, so your data is reproducible.
    """

    '''LLM model itself'''
    llmmodel = LocalHFJudge()
    '''Stating domains'''
    domains = ["travel","education","computer_science","fashion", "technology", "art","health and wellness","finance","food and cooking","career","technology","shopping",
    "entertainment","writing","science","scheduling","fitness","home improvement", "history", "education"]

    #classes = ["ambiguous"] * (n // 2) + ["clear"] * (n // 2)
    classes = (["ambiguous"] * (n // 2)+ ["clear"] * (n - n // 2))
    seen_prompts = set()
    pairs = []


    for request_class in classes:
        'Generates user request prompt'
        for attempt in range(5):
            #domain = random.choice(domains)
            rng = random.Random(DATA_SEED)
            domain = rng.choice(domains)

            prompt = llmmodel.generate_request(
                request_class, domain
            )

            if prompt and prompt.lower() not in seen_prompts:
                break
        else:
            raise RuntimeError(
                "Could not generate a unique prompt after 5 attempts"
            )

        '''User request storage'''
        seen_prompts.add(prompt.lower())
        '''Generating responses to that prompt'''
        clarification, direct_answer = llmmodel.generate_responses(
            prompt
        )
        '''If the response was amb, chosen is clarification. If the response was clear, chosen is direct answer'''
        if request_class == "ambiguous":
            chosen = clarification
            rejected = direct_answer
        else:
            chosen = direct_answer
            rejected = clarification

        '''Putting it in the correct format'''

        pairs.append({
            "class": request_class,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

        print(f"Generated {len(pairs)}/{n} pairs")

    return pairs


# --- Wired: everything below is provided -------------------------------------------


def write_pairs(pairs: list[dict], path: str | Path = PREFERENCE_DATA_PATH) -> None:
    """Validates, assigns stable ids, and writes pairs as JSONL."""
    required = {"class", "prompt", "chosen", "rejected"}

    for i, p in enumerate(pairs):
        missing = required - p.keys()

        if missing:
            raise ValueError(f"pair #{i} missing keys {sorted(missing)}")

        if p["class"] not in VALID_CLASSES:
            raise ValueError(f"pair #{i} has invalid class {p['class']!r}")

    if len(pairs) < MIN_PREFERENCE_PAIRS:
        raise ValueError(
            f"Only {len(pairs)} pairs; the assignment requires at least " f"{MIN_PREFERENCE_PAIRS}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            rec = {
                "id": f"pair_{i:06d}",
                "class": p["class"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_amb = sum(1 for p in pairs if p["class"] == "ambiguous")
    n_clr = sum(1 for p in pairs if p["class"] == "clear")
    print(f"Wrote {len(pairs)} pairs ({n_amb} ambiguous, {n_clr} clear) -> {path}")


def sample_for_verification(
    pairs_path: str | Path = PREFERENCE_DATA_PATH,
    out_path: str | Path = HAND_VERIFIED_PATH,
    k: int = HAND_VERIFY_SAMPLE,
    seed: int = DATA_SEED,
) -> None:
    """Samples k pairs into a hand-verification file with a blank 'verdict' field.

    Open the file and set each pair's "verdict" to "accept" or "reject" by hand
    (a pair is a reject if the chosen/rejected labels are wrong, the prompt is
    mislabeled ambiguous/clear, or either response is low quality). Then run
    --report.
    """
    pairs = [
        json.loads(l)
        for l in Path(pairs_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rng = random.Random(seed)
    sample = rng.sample(pairs, min(k, len(pairs)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in sample:
            rec = dict(rec)
            rec["verdict"] = ""  # Fill with "accept" or "reject" by hand.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} pairs to verify -> {out_path}")
    print("Fill in each 'verdict' (accept/reject) by hand, then run --report.")


def rejection_rate(path: str | Path = HAND_VERIFIED_PATH) -> float:
    """Computes the rejection rate over the hand-verified sample."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    verdicts = [str(r.get("verdict", "")).strip().lower() for r in rows]

    unfilled = [i for i, v in enumerate(verdicts) if v not in {"accept", "reject"}]
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} of {len(rows)} verdicts are not 'accept'/'reject' "
            f"(first at row {unfilled[0]}). Finish hand-verifying before reporting."
        )

    n_reject = sum(1 for v in verdicts if v == "reject")
    rate = n_reject / len(rows)

    print(f"Hand-verified {len(rows)} pairs: {n_reject} rejected -> rejection rate {rate:.1%}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="synthesize + write pairs, then sample for verification",
    )

    parser.add_argument(
        "--n", type=int, default=MIN_PREFERENCE_PAIRS, help="how many pairs to synthesize"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compute the rejection rate from the hand-verified file",
    )

    args = parser.parse_args()

    if args.generate:
        pairs = synthesize_pairs(args.n)
        write_pairs(pairs)
        sample_for_verification()

    if args.report:
        rejection_rate()

    if not (args.generate or args.report):
        parser.print_help()


if __name__ == "__main__":
    main()
