import torch
import torch.nn as nn
import torch.nn.functional as F

################################################## Constructor blocks ################################################## 
class BackboneModule(nn.Module):
    def __init__(self, module: nn.Module):
        super(BackboneModule, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)

class ScaffoldModule(nn.Module):
    def __init__(self, module: nn.Module, target_head: nn.Module):
        super(ScaffoldModule, self).__init__()
        self.module = module
        self.target_head = target_head

    def forward(self, x):
        x = self.module(x)
        return x, self.target_head(x)

class HeadModule(nn.Module):
    def __init__(self, module: nn.Module, target_head: nn.Module):
        super(HeadModule, self).__init__()
        self.module = module
        self.target_head = target_head

    def forward(self, x):
        return self.target_head(self.module(x))

###################################################### MLP blocks ###################################################### 
class MLPBackbone(BackboneModule):
    def __init__(self, input_size, hidden_sizes, dropout_rate, hidden_activation):        
        if len(hidden_sizes) == 0:
            layers = [nn.Identity()]

        else:
            layers = [nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                hidden_activation,
                nn.Dropout(dropout_rate),
            )]
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    hidden_activation,
                    nn.Dropout(dropout_rate),
                ))          
        
        super(MLPBackbone, self).__init__(nn.Sequential(*layers))

class MLPScaffoldModule(ScaffoldModule):
    def __init__(self, input_size, output_size, dropout_rate, hidden_activation):
        module = nn.Sequential(
            nn.Linear(input_size, output_size),
            hidden_activation,
            nn.Dropout(dropout_rate),
        )
        target_head = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid(),
        )
        super(MLPScaffoldModule, self).__init__(module, target_head)

class BinaryClassificationHead(HeadModule):
    def __init__(self, input_size, hidden_size, dropout_rate, hidden_activation):
        module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            hidden_activation,
            nn.Dropout(dropout_rate),
        )
        target_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        super(BinaryClassificationHead, self).__init__(module, target_head)

###################################################### VGG blocks ###################################################### 
_VGG16_BN_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]

def _make_conv_layers(cfg, in_channels: int = 3) -> nn.Sequential:
    layers= []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            v = int(v)
            layers.extend(
                [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = v
    return nn.Sequential(*layers)

class VGGBackbone(BackboneModule):
    def __init__(self, type: str = "vgg16", dropout: float = 0.5):
        # > Input validation
        if type not in ["vgg16", "vgg19"]:
            raise ValueError("Invalid VGG type")
        if type == "vgg19":
            raise ValueError("VGG19 is not supported yet")

        features = nn.Sequential(
            _make_conv_layers(_VGG16_BN_CFG, in_channels=3),
            nn.Identity(),
            nn.Flatten(),
        )

        super(VGGBackbone, self).__init__(features)

class VGGScaffoldModule(ScaffoldModule):
    def __init__(self, n_classes, dropout: float = 0.5):
        module = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        target_head = nn.Linear(4096, n_classes)

        super(VGGScaffoldModule, self).__init__(module, target_head)

class VGGMultiClassClassificationHead(HeadModule):
    def __init__(self, n_classes, dropout: float = 0.5):
        module = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),            
        )
        target_head = nn.Linear(4096, n_classes)
        super(VGGMultiClassClassificationHead, self).__init__(module, target_head)

###################################################### GPT blocks ###################################################### 
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):    
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos) 

        return tok_emb + pos_emb

class GPTBackbone(BackboneModule):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_blocks, dropout):
        token_and_position_embedding = TokenAndPositionEmbedding(vocab_size, block_size, n_embd)
        blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_blocks)])

        super(GPTBackbone, self).__init__(nn.Sequential(token_and_position_embedding, blocks))

class GPTScaffoldModule(ScaffoldModule):
    def __init__(self, n_embd, n_head, n_classes, block_size, dropout):
        module = Block(n_embd, n_head, block_size, dropout)
        target_head = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, n_classes))
        
        super(GPTScaffoldModule, self).__init__(module, target_head)

class GPTMultiClassClassificationHead(HeadModule):    
    def __init__(self, n_embd, n_head, n_classes, block_size, dropout):
        module = Block(n_embd, n_head, block_size, dropout)
        target_head = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, n_classes))
        
        super(GPTMultiClassClassificationHead, self).__init__(module, target_head)