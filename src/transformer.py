from tinygrad.tensor import Tensor
import math
import functools

# Implementation from attention is all you need https://arxiv.org/pdf/1706.03762, with dropout and prenorm
# And https://github.com/tinygrad/tinygrad/blob/24dd0d52edfc32ab6f887f22752145255d8524dc/extra/models/transformer.py#L5


# NOTE: depending on how do you do the operations (like doing operations by using intermediary variables for reading), tinygrad can generate a kernel that surpasses the amount of buffers a metal kernel can have (Metal can only have 31, buffers so buffer 31 is already out of bounds). Supposedly it was fixed (https://github.com/tinygrad/tinygrad/pull/10510) but this version still doesn't have a release


class TransformerBlock:
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: int = 0.1):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.key_dimension = (
            embed_dim // num_heads
        )  # value dimension = key dimension here (head size)
        self.dropout = dropout

        self.query = (
            Tensor.scaled_uniform(embed_dim, embed_dim),
            Tensor.zeros(embed_dim),
        )
        self.key = (
            Tensor.scaled_uniform(embed_dim, embed_dim),
            Tensor.zeros(embed_dim),
        )
        self.value = (
            Tensor.scaled_uniform(embed_dim, embed_dim),
            Tensor.zeros(embed_dim),
        )

        self.out = (
            Tensor.scaled_uniform(embed_dim, embed_dim),
            Tensor.zeros(embed_dim),
        )

        self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
        self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

        self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
        self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

    def attn(self, x: Tensor, attn_mask: Tensor | None = None):
        """
        Multi-head scaled-dot-product attention.

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (batch, seq_len, embed_dim).
        attn_mask : Tensor or None, optional
            Boolean mask of shape (batch, seq_len, seq_len).
            Positions that are ``False`` will be masked out (set to -inf)
            before the softmax.  If ``None``, no masking is applied.

        Returns
        -------
        Tensor
            Output tensor of shape  (batch, seq_len, embed_dim)
        """
        query = x.linear(*self.query)  # (batch, seq_len, embed_dim)
        query = query.reshape(
            query.shape[0], query.shape[1], self.num_heads, self.key_dimension
        ).transpose(1, 2)  #  (batch, heads, seq_len, query_dimension)
        key = x.linear(*self.key)  #  (batch, seq_len, embed_dim)
        key = key.reshape(
            key.shape[0], key.shape[1], self.num_heads, self.key_dimension
        ).transpose(1, 2)  # (batch, heads, seq_len, key_dimension)
        value = x.linear(*self.value)  # (batch, seq_len, embed_dim)
        value = value.reshape(
            value.shape[0], value.shape[1], self.num_heads, self.key_dimension
        ).transpose(1, 2)  # (batch, heads, seq_len, key_dimension)

        # Scaled dot product
        qk = (
            query.matmul(key.transpose(-1, -2)) * (1 / math.sqrt(self.key_dimension))
        )  # (batch, heads, seq_len, seq_len)

        if attn_mask is not None:
            # this in a way is masked fill, where here the values we don't want we are going to add to them infinity
            attn_mask = attn_mask.where(0, -float("inf"))
            # attn_mask = attn_mask.where(1, 0) # because the sum of 1s will get cancelled in the division because we will have e^{x + 1} = e^x * e^1
            qk = qk + attn_mask.reshape(1, 1, attn_mask.shape[0], attn_mask.shape[1])

        attn_score = (
            qk.softmax(axis=-1) @ value
        )  # (batch, heads, seq_len, key_dimension)

        # we concatenate the heads and mix their represenations (we learn a re-weighting of this heads)
        attn_score = attn_score.transpose(1, 2).reshape(
            -1, x.shape[1], self.num_heads * self.key_dimension
        ).linear(*self.out)  # (batch, seq_len, embed_dim)

        return attn_score

    def __call__(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        Multi-head scaled-dot-product attention.

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (batch, seq_len, embed_dim).
        attn_mask : Tensor or None, optional
            Boolean mask of shape (batch, seq_len, seq_len).
            Positions that are ``False`` will be masked out (set to -inf)
            before the softmax.  If ``None``, no masking is applied.

        Returns
        -------
        Tensor
            Output tensor of shape (batch, heads, seq_len, key_dimension).
        """

        # pre-norm
        attn_score = self.attn(x.layernorm().linear(*self.ln1), attn_mask).dropout(self.dropout)
        x = x + attn_score

        # ff
        x = x.layernorm().linear(*self.ln2)  # pre-norm
        ff_out_1 = x.linear(*self.ff1).relu()
        ff_out_2 = ff_out_1.linear(*self.ff2).dropout(self.dropout)

        # residual
        x = x + ff_out_2

        return x  # (batch, seq_len, embed_dim)


class DecoderTransformer:
    def __init__(
        self, max_len: int, vocab_dim: int, embed_dim: int, num_heads: int, layers: int, ff_dim: int
    ):
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)
        ]

        self.pos_embed = Tensor.scaled_uniform(max_len, embed_dim)
        self.embedder = Tensor.scaled_uniform(vocab_dim, embed_dim)
        self.proj_ff = Tensor.scaled_uniform(embed_dim, vocab_dim)

    def forward(self, x: Tensor, logits_only: bool = True):
        """
        Multi-head scaled-dot-product attention.

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (batch, seq_len, 1).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, seq_len, vocab_dim).
        """

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        onehot_feat = x.int().one_hot(self.vocab_dim)

        x = onehot_feat.dot(self.embedder)  # (batch, seq_len, embed_dim)
        x = x + self.pos_embed[:seq_len, :]

        attn_mask = Tensor.ones(x.shape[1], x.shape[1]).tril()

        x = functools.reduce(lambda x, f: f(x, attn_mask), self.transformer_blocks, x)

        x = x.dot(self.proj_ff)  # (batch, seq_len, vocab_dim)

        if logits_only: # Non tinygrad operations are not supported by jit
            return x

        return x.softmax(axis=-1)  # (batch, seq_len, vocab_dim)

    def generate(
        self, sequence: Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample: bool=False, top_k: int|None=None
    ) -> Tensor:
        """
        Take a conditioning sequence of indices sequence (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (batch, seq_len).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, seq_len).
        """
        temp = Tensor.training
        Tensor.training = False

        for _ in range(max_new_tokens):
            sequence_cond = (
                sequence
                if sequence.shape[1] <= self.max_len
                else sequence[:, -self.max_len :]
            )

            logits = self.forward(sequence_cond, True)

            if top_k is not None:
                v, _ = logits.topk(top_k)
                logits = (logits < v[:, :, -1].unsqueeze(-1)).where(-float("inf"), logits)

            probs = (logits[:, -1, :] / temperature).softmax(
                axis=-1
            )  # we only want the last token of the generation (the target)

            if do_sample:
                idx_next = probs.multinomial(num_samples=1)
            else:
                idx_next = probs.argmax(axis=-1).unsqueeze(-1)

            sequence = Tensor.cat(sequence, idx_next, dim=-1)

        Tensor.training = temp

        return sequence


class EncoderTransformer:
    def __init__(
        self, max_len: int, vocab_dim: int, embed_dim: int, num_heads: int, layers: int, ff_dim: int
    ):
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)
        ]

        self.pos_embed = Tensor.scaled_uniform(max_len, embed_dim)
        self.embedder = Tensor.scaled_uniform(vocab_dim, embed_dim)
        self.proj_ff = Tensor.scaled_uniform(embed_dim, vocab_dim)

    def forward(self, x: Tensor, logits_only: bool = True):
        """
        Multi-head scaled-dot-product attention.

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (batch, seq_len, 1).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, seq_len, vocab_dim).
        """

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        onehot_feat = x.int().one_hot(self.vocab_dim)

        x = onehot_feat.dot(self.embedder)  # (batch, seq_len, embed_dim)
        x = x + self.pos_embed[:seq_len, :]

        x = functools.reduce(lambda x, f: f(x, None), self.transformer_blocks, x)

        x = x.dot(self.proj_ff)  # (batch, seq_len, vocab_dim)

        if logits_only:
            return x

        return x.softmax(axis=-1)  # (batch, seq_len, vocab_dim)
