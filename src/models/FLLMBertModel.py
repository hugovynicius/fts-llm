# from flautim.pytorch.Model import Model
from torch import nn
from transformers import DistilBertModel
from peft import LoraConfig, TaskType, get_peft_model
from src.models.AttentionPooling import AttentionPooling

class FLLMBertModel(nn.Module):
    def __init__(self, context, output_size, tokenizer, hidden_dims=[128, 64], use_lora=True, **kwargs):
        kwargs.pop("name", None)
        super().__init__()

        self.lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                            inference_mode=False,
                            r=64,
                            lora_alpha=128,
                            lora_dropout=0.1,
                            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"] # DistilBERT-specific modules
        )

        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.resize_token_embeddings(len(tokenizer))
        self.output_size = output_size
        self.use_lora = use_lora

        for param in self.model.embeddings.word_embeddings.parameters():
            param.requires_grad = True

        if self.use_lora:
            self.model = get_peft_model(self.model, self.lora_config)

        # Attention-based pooling over the hidden states
        self.attn_pool = AttentionPooling(self.model.config.hidden_size)

        # MLP head for forecasting
        layers = []
        in_dim = self.model.config.hidden_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels=None):

        model_outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = model_outputs.last_hidden_state
        pooled = self.attn_pool(hidden_states)
        output = self.mlp_head(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, labels)
        else:
            loss = None

        return {"loss" : loss, "logits" : output}