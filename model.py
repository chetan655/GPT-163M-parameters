import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.shift = nn.Parameter(torch.zeros(cfg["emb_dim"]))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = ( x - mean) / torch.sqrt( var + self.eps)
        return norm_x * self.scale + self.shift
    

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        assert (cfg["emb_dim"] % cfg["n_heads"] == 0), "emb_dim must be divisible by n_heads"
        self.n_head = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // self.n_head

        self.w_query = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.w_key = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.w_value = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])

        self.dropout = nn.Dropout(cfg["dropout"])
        self.register_buffer("mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1))

    def forward(self, x):
        batch, n_tokens, emb_dim = x.shape
        '''
            x -> (batch, n_tokens, emb_dim)
            w_key/query/value -> (emb_dim, emb_dim)
            self.w_key/query/value(x) -> (batch, n_tokens, emb_dim)
        '''
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        keys = keys.view(batch, n_tokens, self.n_head, self.head_dim)
        queries = queries.view(batch, n_tokens, self.n_head, self.head_dim)
        values = values.view(batch, n_tokens, self.n_head, self.head_dim)

        keys = keys.transpose(1, 2)   # (batch, n_head, n_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) # (batch, n_heads, n_tokens, n_tokens)

        mask = self.mask[:n_tokens, :n_tokens]

        attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )

        attn_weights = self.dropout(attn_weights)
        
        context_vector = attn_weights @ values   # (batch, n_heads, n_tokens, head_dim)

        context_vector = context_vector.transpose(1, 2) # (batch, n_tokens, n_heads, head_dim)

        context_vector = context_vector.contiguous().view(batch, n_tokens, self.n_head * self.head_dim)  # (batch, n_tokens, emb_dim)

        context_vector = self.out(context_vector)

        return context_vector

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer1 = nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4, bias=cfg["qkv_bias"])
        self.layer2 = nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.gelu = GELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layerNorm1 = LayerNorm(cfg)
        self.layerNorm2 = LayerNorm(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.feedForward = FeedForward(cfg)
        self.multiHeadAttention = MultiHeadAttention(cfg)

    def forward(self, x):
        shortcut = x
        x = self.layerNorm1(x)
        x = self.multiHeadAttention(x)
        x = self.dropout(x)

        x = x + shortcut

        shortcut = x
        x = self.layerNorm2(x)
        x = self.feedForward(x)
        x = self.dropout(x)

        x = x + shortcut

        return x
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

        self.tf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.norm = LayerNorm(cfg)
        self.proj_out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, input):
        batch, seq_len = input.shape

        tok_emb = self.token_embedding(input)
        pos_emb = self.pos_embedding(torch.arange(seq_len, device=input.device))
        input_embedding = tok_emb + pos_emb

        x = self.dropout(input_embedding)
        x = self.tf_blocks(x)
        x = self.norm(x)

        '''
            x -> (batch, seq_len, emb_dim)
            proj_out -> (emb_dim, vocab_size)
            result -> (batch, seq_len, vocab_size)
            or we can say for each token we get probabilities of all the vocab
        '''
        logits = self.proj_out(x)  
        return logits




