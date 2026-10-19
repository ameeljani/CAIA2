"""
Part A -- LoRA SFT training entrypoint.

SFT warm-starts the target behavior: it fine-tunes the instruct base model on the
PREFERRED response of each preference pair you built (a clarifying question for an
ambiguous prompt, a direct answer for a clear one), so that DPO in Part B has a sensible
policy to sharpen rather than starting cold.

CLI args, tokenizer setup, base model + LoRA construction, and checkpointing are wired
up already. You implement utils/masking_utils.tokenize_with_response_mask() and
train_loop() -- see the TODO blocks and README.md's Part A for two gotchas: freezing
base weights, masking prompt tokens.
"""

import argparse

import torch
from peft import LoraConfig, get_peft_model

from configs.paths_config import SFT_CHECKPOINT_DIR
from configs.train_config import (
    MODEL_ID,
    RANK,
    ALPHA,
    DROPOUT,
    TARGET_MODULES,
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    SFT_MAX_SEQ_LEN,
)
from data_loader import load_sft_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def build_model_and_tokenizer():
    """Loads the base model + tokenizer, and wraps the model with a LoRA adapter."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model()

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Sanity check: should be a small fraction of total params.

    return model, tokenizer


def build_sft_example(messages: list[dict], tokenizer, max_seq_len: int = SFT_MAX_SEQ_LEN) -> dict:
    """
    Wired up: converts utils.masking_utils.tokenize_with_response_mask()'s response_mask into
    the -100-masked labels tensor the causal-LM loss needs. Once you implement
    tokenize_with_response_mask(), this works for free -- nothing to implement here.

    Returns a dict with "input_ids", "attention_mask", "labels": labels equal
    input_ids where response_mask == 1 (the final assistant turn), -100 elsewhere.
    """
    example = tokenize_with_response_mask(messages, tokenizer, max_seq_len)

    labels = example["input_ids"].clone()
    labels[example["response_mask"] == 0] = -100

    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": labels,
    }




def train_loop(model, tokenizer, dataset) -> None:
    """
    Implements the Part A SFT training loop.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_sft_example() per example
         (e.g. via `dataset.map(...)`) to get input_ids/attention_mask/labels -- do not
         call build_sft_example() inside the per-step loop below, or you'll re-tokenize
         the same examples on every epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate into a batch, forward pass,
         compute loss, backward, optimizer step. Only LoRA parameters should have
         gradients -- true by default from build_model_and_tokenizer() above, but verify
         it if you change the LoRA setup (README.md Part A, gotcha #1).
      3. Log the loss periodically so you can sanity-check training is converging.

    Hyperparameters (SFT_BATCH_SIZE, SFT_EPOCHS, SFT_LEARNING_RATE, SFT_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune, not fixed by this skeleton.
    """

    '''Collator function'''

    def collate_fn(batch):
        input_ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True,
                                 padding_value=tokenizer.pad_token_id)

        attention_mask = pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0
        )

        labels = pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    '''check tokenizer'''
    #tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    '''preprocessing loop'''
    tokenized_dataset = dataset.map(
        lambda example: build_sft_example(example['messages'], tokenizer),
        remove_columns=dataset.column_names
    )
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    '''split into batches'''
    batch_size = SFT_BATCH_SIZE
    learning_rate = SFT_LEARNING_RATE

    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )




    '''init optimizer'''
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate)

    '''train loop'''
    epochs = SFT_EPOCHS
    epochs_store = []
    losses_store = []
    model.train()
    '''device'''
    device = model.get_input_embeddings().weight.device
    for epoch in range(epochs):
        epochs_store.append(epoch)
        '''go through batches'''
        running_loss = 0.0
        print("Epoch" + str(epoch), flush = True)
        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            optimizer.zero_grad()

            '''forward pass '''
            outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])

            '''compute loss, evidenly this is already built in as cross entropy'''
            loss = outputs.loss

            '''record loss'''
            running_loss += loss.item()

            '''backward pass'''
            loss.backward()


            '''optimizer step'''
            optimizer.step()

            '''Printing out status for batch training'''
            if (step + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{SFT_EPOCHS}, "f"Step {step + 1}/{len(train_loader)}, "f"Loss: {loss.item():.4f}")

        '''printing out status for the epoch itself'''
        avg_epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1} completed. "
            f"Average Loss: {avg_epoch_loss:.4f}", flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading base model {MODEL_ID} and building LoRA adapter...")
    model, tokenizer = build_model_and_tokenizer()

    print("Loading SFT examples (preferred responses from your preference data)...")
    dataset = load_sft_examples()

    print("Training starts...")
    train_loop(model, tokenizer, dataset)

    print(f"Saving LoRA adapter to {args.checkpoint_dir}...")
    model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)
    print("Part A SFT complete.")


if __name__ == "__main__":
    main()








