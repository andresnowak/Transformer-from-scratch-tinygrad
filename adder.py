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


# dataset idea from https://github.com/karpathy/minGPT/blob/master/projects/adder/adder.py
def make_dataset():
    ds = []
    for i in range(100):
        for j in range(100):
            s = i + j
            ds.append(
                [i // 10, i % 10, j // 10, j % 10, s // 100, (s // 10) % 10, s % 10]
            )
    random.shuffle(ds)
    ds = np.array(ds).astype(np.float32)
    ds_X = ds[:, 0:6]
    ds_Y = np.copy(ds[:, 1:]).astype(np.int32)
    ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
    ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]
    return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test


class AdderDataset:
    def __init__(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
    ):
        self.x = x
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(len(self.x))

    def __reset__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Return the iterator object (self)."""
        self.__reset__()
        self._current = 0
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        """Return the next mini-batch."""
        if self._current >= len(self.x):
            raise StopIteration

        # Slice indices for the current batch
        idx = self._indices[self._current : self._current + self.batch_size]
        x_batch = self.x[idx]
        y_batch = self.y[idx]

        self._current += self.batch_size
        return Tensor(x_batch), Tensor(y_batch)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a single sample (x, y) by index."""
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.batch_size))



def evalauate(model: DecoderTransformer, test_dataset: AdderDataset, return_predict: bool=False) -> Union[float, Tuple[float, np.ndarray]]:
    Tensor.training = False

    correct = 0
    total = 0
    all_preds = []
    for x_batch, y_batch in test_dataset:
        out = model.forward(x_batch, logits_only=False)

        y_test_preds = np.argmax(out.numpy(), axis=-1)
        correct += (y_batch.numpy() == y_test_preds).sum()
        total += y_batch.numpy().size
        all_preds.append(y_test_preds)

        for k in range(len(x_batch)):
            if y_batch.numpy()[k, -3:].astype(np.int32)[0] == y_test_preds[k, -3:].astype(np.int32)[0]:
                a,b,c,x = x_batch.numpy()[k, :2].astype(np.int32), x_batch.numpy()[k, 2:4].astype(np.int32), y_batch.numpy()[k, -3:].astype(np.int32), y_test_preds[k, -3:].astype(np.int32)
                print(f'{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})')

    acc = (correct / total) if total > 0 else 0.0
    all_preds = np.concatenate(all_preds, axis=0)

    print(f"test set accuracy is {acc:.6f}")

    return (acc, all_preds) if return_predict else acc


if __name__ == "__main__":
    lr = 3e-4
    epochs = 60
    batch_size = 64
    seq_len = 6

    model = DecoderTransformer(max_len=seq_len, vocab_dim=10, embed_dim=128, num_heads=4, layers=3, ff_dim=128 * 4, use_MoE=None)

    print(len(get_parameters(model)))

    X_train, Y_train, X_test, Y_test = make_dataset()
    train_dataset = AdderDataset(X_train, Y_train, batch_size)
    test_dataset = AdderDataset(X_test, Y_test, batch_size, False)

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
    train_step = TinyJit(train_step)

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
                    loss, accuracy = train_step(x_batch, y_batch)
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

    acc, preds = evalauate(model, test_dataset, True)
