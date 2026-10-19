"""
Shared tokenization + response-masking logic used by both Part A (train_sft.py) and
Part B (train_dpo.py). "Mask everything except the final assistant turn" is the same
operation in both places, so it lives here once instead of being re-derived
independently in two files.
"""

import torch


def tokenize_with_response_mask(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    """
    TODO (student): tokenizes one chat-formatted conversation and marks which tokens
    belong to the final assistant turn. `messages` is a list of {"role", "content"}
    dicts (user/assistant, alternating, ending on an assistant turn) -- see data_loader.py.

    Must return a dict with:
      - "input_ids": 1D LongTensor -- the full tokenized conversation
        (`tokenizer.apply_chat_template(messages, tokenize=True, ...)`), truncated to
        at most max_seq_len tokens.

      - "attention_mask": 1D LongTensor, same length, all 1s (no padding needed here --
        batching/collating happens in the caller's train_loop()).

      - "response_mask": 1D LongTensor, same length: 1 for the final assistant-turn
        tokens, 0 everywhere else (prompt tokens and any earlier turns).

    This is the function the autograder unit-tests directly (see README.md's
    Autograder section) -- its signature and the three return keys above must not
    change. Both train_sft.build_sft_example() and train_dpo.build_dpo_example() call
    this rather than re-deriving the masking logic independently -- implement it once,
    here, and both Part A and Part B get it for free.

    Hint: `tokenizer.apply_chat_template(messages[:-1], tokenize=True,
    add_generation_prompt=True)` tokenizes everything up through the prompt that
    invites the final assistant turn. Its length tells you exactly where the response
    starts in the full conversation's tokenization -- everything before that index is
    prompt, everything from it onward (up to the end-of-turn token) is response.
    """

    '''full tokenized conversation'''
    text_all = tokenizer.apply_chat_template(messages, tokenize=True,add_generation_prompt=False)
    input_text = torch.tensor(text_all[:max_seq_len],dtype=torch.long)
    input_length = len(input_text)

    '''Debugging stmt'''
    input_list = input_text.tolist()

    '''attention mask'''
    tensor_ones = torch.ones(input_length, dtype=torch.long)

    '''Debugging stmt'''
    attnmask_list = tensor_ones.tolist()

    '''response mask'''
    text_excludingprompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True,add_generation_prompt=True)
    where_response_starts = len(text_excludingprompt)
    resp_mask = torch.zeros(input_length, dtype=torch.long)
    resp_mask[where_response_starts:] = 1

    '''Debugging stmt'''
    respmask_list = resp_mask.tolist()

    '''Output'''
    output = {'input_ids': input_text, 'attention_mask': tensor_ones, 'response_mask': resp_mask}
    return output










