import tempfile

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate", reason="accelerate is required by Trainer")
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments


class TinyCausalDataset(Dataset):
    """Synthetic dataset for quick trainer smoke tests."""

    def __init__(self, vocab_size=128, seq_len=16, length=8):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.length = length
        self.inputs = torch.randint(0, vocab_size, (length, seq_len))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ids = self.inputs[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def test_tiny_model_forward():
    """Ensure a minimal GPT-2 config runs a forward pass."""
    config = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape[:2] == (1, 8)


def test_trainer_single_step():
    """Run a single training step on synthetic data to validate the training loop."""
    config = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    dataset = TinyCausalDataset(vocab_size=config.vocab_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir,
            overwrite_output_dir=True,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            max_steps=1,
            logging_steps=1,
            report_to="none",
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
