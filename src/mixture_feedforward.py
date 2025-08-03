from tinygrad.tensor import Tensor


class TopKGate:
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 1):
        self.top_k = top_k

        self.gate_linear = Tensor.scaled_uniform(
            input_dim, num_experts
        )  # this layer computes the logits of the experts, which then basically get converted to the weights with softmax (as the combination of the experts is just a weighted sum)

    def __call__(self, x: Tensor):
        """
        x: Tensor
            shape (Batch * seq_len, embed_dim)
        """

        logits = x.dot(self.gate_linear)  # (Batch * seq_len, num_experts)

        top_k_logits, top_k_indices = logits.topk(
            self.top_k, dim=-1
        )  # (Batch * seq_len, k)

        top_k_weights = top_k_logits.softmax(axis=-1)  # (Batch * seq_len, k)

        full_weights = Tensor.zeros_like(logits)  # (Batch * seq_len, num_experts)
        full_weights = full_weights.scatter(
            1, top_k_indices, top_k_weights
        )  # (Batch * seq_len, num_experts)

        return full_weights, top_k_indices


class MoELayer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        top_k=1,
        hidden_dim: int | None = None,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim

        if hidden_dim is None:
            hidden_dim = input_dim * 4  # Also from the Attention is all you need paper

        self.gate = TopKGate(input_dim, num_experts, top_k)

        self.ff_1_experts = Tensor.scaled_uniform(num_experts, input_dim, hidden_dim)
        self.ff_2_experts = Tensor.scaled_uniform(num_experts, hidden_dim, output_dim)

    def __call__(self, x: Tensor):
        """
        x: Tensor
            (Batch, seq_len, embed_dim)
        """
        B, S, D = x.shape
        B, S, D = int(B), int(S), int(D)
        x_flat = x.reshape(-1, D)  # (Batch * seq_len, input_dim("embed_dim"))

        gate_weights, k_indices = self.gate(x_flat)  # probs, indices

        # MoE without loop (but not the most efficient as we do complete operation and not sparse)
        # Because gates is already 0 where expert is not activated we can do an einsum
        h = x_flat.reshape(B * S, 1, 1, D).dot(self.ff_1_experts).relu()

        # we reshape to do (B*S, 1, 1, D) @ (E, D, H), basically we will do 1, D @ D, H E times and then B*S times, because we need to do it for every expert and each position in B*S
        o = h.dot(self.ff_2_experts).reshape(
            B * S, self.num_experts, D
        )  # (Batch * seq_len, Experts, embed_dim)

        out = (gate_weights.unsqueeze(-1) * o).transpose(1, 2).sum(axis=-1)

        return out.reshape(B, S, D)
