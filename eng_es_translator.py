#!/usr/bin/env python3
import numpy as np
import random
from typing import Iterator, Tuple, Union
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset
import torch
from tinygrad.dtype import _from_torch_dtype

from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit

from src.transformer import EncoderDecoderTransformer

device = "METAL"


class TranslatorDataset:
    def __init__(
        self,
        data: Dataset,
        batch_size: int = 32,
        seq_len: int = 128,
        shuffle: bool = True,
    ):
        SPECIALS = {
            "@": -100,
            "^": 0,
        }  # PAD and BOS tokens (this chars don't appear in the dataset)

        chars = set()
        for lang in ["English", "Spanish"]:
            for row in data[lang]:
                chars.update(row)
        chars = sorted(chars)

        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} and vocab size is {vocab_size}")

        self.stoi = SPECIALS.copy()
        self.stoi.update({c: i + len(SPECIALS) - 1 for i, c in enumerate(chars)})
        self.itos = {v: k for k, v in self.stoi.items()}

        self.vocab_size = vocab_size

        def encode_data(row):
            eng_text = [self.stoi["^"]] + [self.stoi[c] for c in row["English"]]
            es_text = [self.stoi["^"]] + [self.stoi[c] for c in row["Spanish"]]

            return {"English": eng_text, "Spanish": es_text}

        self.data = data.map(encode_data, remove_columns=data.column_names)
        self.seq_len = seq_len

        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(
            0, len(self.data) - self.seq_len
        )  # last char has no target

    def __reset__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._current = 0

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        """Return the iterator object (self)."""
        self.__reset__()
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the next mini-batch."""
        if self._current + self.batch_size > len(self._indices):
            raise StopIteration

        # Slice indices for the current batch
        idxs = self._indices[self._current : self._current + self.batch_size]

        enc = self.data["English"][idxs]
        dec = self.data["Spanish"][idxs]

        enc = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in enc],
            batch_first=True,
            padding_value=self.stoi["@"],
        )
        enc = torch.nn.functional.pad(
            enc, (0, self.seq_len - enc.size(1)), value=self.stoi["@"]
        )

        dec = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in dec],
            batch_first=True,
            padding_value=self.stoi["@"],
        )
        dec = torch.nn.functional.pad(
            dec, (0, self.seq_len + 1 - dec.size(1)), value=self.stoi["@"]
        )

        target = dec[:, 1:]
        dec = dec[:, :-1]

        enc = Tensor.from_blob(enc.data_ptr(), enc.shape, dtype=_from_torch_dtype(enc.dtype), device=str(enc.device)).to(device)
        dec = Tensor.from_blob(dec.data_ptr(), dec.shape, dtype=_from_torch_dtype(dec.dtype), device=str(dec.device)).to(device)
        target = Tensor.from_blob(target.data_ptr(), target.shape, dtype=_from_torch_dtype(target.dtype), device=str(target.device)).to(device)

        self._current += self.batch_size
        return enc, dec, target

    def __len__(self) -> int:
        return int(np.ceil((len(self.data) - self.seq_len) / self.batch_size))


if __name__ == "__main__":
    lr = 5e-4
    epochs = 3
    batch_size = 64
    seq_len = 128

    def filter(row):
        if row["English"] != None and row["Spanish"] != None and len(row["English"]) < seq_len-1 and len(row["Spanish"]) < seq_len-1:
            return row

    dataset: Dataset = load_dataset(
        "okezieowen/english_to_spanish",
        split="train",
    )
    dataset = dataset.filter(filter)

    train_dataset = TranslatorDataset(dataset, batch_size, seq_len)

    model = EncoderDecoderTransformer(
        max_len=seq_len,
        vocab_dim=train_dataset.vocab_size,
        embed_dim=128,
        num_heads=8,
        layers=4,
        ff_dim=128 * 4,
    )

    optim = Adam(get_parameters(model), lr=lr)

    def loss_fn(out: Tensor, y: Tensor):
        return out.sparse_categorical_crossentropy(y, ignore_index=-100)

    def train_step(x_enc: Tensor, x_dec: Tensor, y: Tensor):
        out = model.forward(x_enc, x_dec, logits_only=True)
        loss = loss_fn(out, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        accuracies = out.argmax(axis=-1) == y  # (batch_size, seq_len)

        return loss.realize(), accuracies.realize()

    train_step_fn = TinyJit(train_step)

    losses, accuracies = [], []

    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracies = 0.0

        with Tensor.train():
            with tqdm(
                train_dataset,
                total=len(train_dataset),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            ) as pbar:
                for step, (x_enc_batch, x_dec_batch, y_batch) in enumerate(pbar):
                    loss, accuracy = train_step_fn(x_enc_batch, x_dec_batch, y_batch)
                    loss, accuracy = loss.numpy(), accuracy.numpy()

                    total_loss += loss
                    total_accuracies += accuracy.sum()

                    pbar.set_postfix(
                        loss=f"{loss}",
                        acc=f"{accuracy.mean()}",
                    )

        # Average over the entire dataset
        losses.append(total_loss / len(train_dataset))
        accuracies.append(
            total_accuracies / (len(train_dataset) * batch_size * seq_len)
        )

        print(f"Loss: {losses[epoch]}, Accuracy: {accuracies[epoch]}")

    generation = model.generate(Tensor([train_dataset.stoi[s] for s in "^Hello how are you"], requires_grad=False).reshape(1, -1), Tensor([train_dataset.stoi[s] for s in "^"], requires_grad=False).reshape(1, -1), 40, do_sample=True, top_k=3)
    print(generation.shape)
    print("".join([train_dataset.itos[s] for s in list(generation.numpy().reshape(-1))]))
