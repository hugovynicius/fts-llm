from torch import nn
from transformers import GPT2Model, GPT2Config
from src.models.AttentionPooling import AttentionPooling


class FLLMGPT2Model(nn.Module):
    def __init__(self, output_size=1, hidden_dims=[128, 64], tokenizer = None):
        super().__init__()
        config = GPT2Config()
        self.gpt2 = GPT2Model(config)
        self.gpt2.resize_token_embeddings(len(tokenizer))
        # Attention-based pooling over the hidden states
        self.attn_pool = AttentionPooling(config.n_embd)

        # MLP head for forecasting
        layers = []
        in_dim = config.n_embd
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels):

        gpt_outputs = self.gpt2(input_ids, attention_mask = attention_mask)
        hidden_states = gpt_outputs.last_hidden_state
        pooled = self.attn_pool(hidden_states)
        output = self.mlp_head(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, labels)


        return {"loss" : loss, "logits" : output.squeeze(0)}
