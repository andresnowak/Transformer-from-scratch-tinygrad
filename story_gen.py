#!/usr/bin/env python3
import numpy as np
import random
from typing import Iterator, Tuple, Union
from tqdm import tqdm

from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit

from src.transformer import DecoderTransformer


class StoryDataset:
    def __init__(
        self, data: str, batch_size: int = 32, seq_len: int = 128, shuffle: bool = True
    ):
        chars = sorted(list(set(data)))
        # Here using a [BOS] token doesn't make sense because we are training from the corpus by grabbing parts of a very big text (so majority of times not beggining of sentences)

        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} and vocab size is {vocab_size}")

        self.stoi = {char: num for num, char in enumerate(chars)}
        self.itos = {num: char for num, char in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data = data
        self.seq_len = seq_len

        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(0, len(self.data) - self.seq_len) # last char has no target

    def __reset__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._current = 0

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Return the iterator object (self)."""
        self.__reset__()
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        """Return the next mini-batch."""
        if self._current + self.batch_size > len(self._indices):
            raise StopIteration

        # Slice indices for the current batch
        idxs = self._indices[self._current : self._current + self.batch_size]

        batch_x, batch_y = [], []
        for start in idxs:
            chunk = self.data[start: start + seq_len + 1]
            dix = [self.stoi[s] for s in chunk]

            batch_x.append(dix[:-1])
            batch_y.append(dix[1:])

        self._current += self.batch_size
        return Tensor(batch_x), Tensor(batch_y)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # grab a chunk of (seq_len + 1) characters from the data
        chunk = self.data[idx:idx + self.seq_len + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = Tensor(dix[:-1])
        y = Tensor(dix[1:])
        return x, y

    def __len__(self) -> int:
        return int(np.ceil((len(self.data) - self.seq_len) / self.batch_size))


if __name__ == "__main__":
    lr = 5e-4
    epochs = 1
    batch_size = 64
    seq_len = 128

    text = open("dataset/shakespear.txt", "r").read()
    train_dataset = StoryDataset(text, batch_size, seq_len)

    model = DecoderTransformer(max_len=seq_len, vocab_dim=train_dataset.vocab_size, embed_dim=128, num_heads=8, layers=4, ff_dim=128 * 4)

    optim = Adam(get_parameters(model), lr=lr)

    def loss_fn(out: Tensor, y: Tensor):
        return out.sparse_categorical_crossentropy(y)

    def train_step(x: Tensor, y: Tensor):
        out = model.forward(x, logits_only=True)
        loss = loss_fn(out, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        accuracies = out.argmax(axis=-1) == y # (batch_size, seq_len)

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
                for step, (x_batch, y_batch) in enumerate(pbar):
                    loss, accuracy = train_step_fn(x_batch, y_batch)
                    loss, accuracy = loss.numpy(), accuracy.numpy()

                    total_loss += loss
                    total_accuracies += accuracy.sum()

                    pbar.set_postfix(
                        loss=f"{loss}",
                        acc=f"{accuracy.mean()}",
                    )

        # Average over the entire dataset
        losses.append(total_loss / len(train_dataset))
        accuracies.append(total_accuracies / (len(train_dataset) * batch_size * seq_len))

        print(f"Loss: {losses[epoch]}, Accuracy: {accuracies[epoch]}")

    generation = model.generate(Tensor([train_dataset.stoi[s] for s in "Would"], requires_grad=False).reshape(1, -1), 32, do_sample=True, top_k=3)
    print(generation.shape)
    print("".join([train_dataset.itos[s] for s in list(generation.numpy().reshape(-1))]))
