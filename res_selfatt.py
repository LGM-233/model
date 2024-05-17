import torch
import torch.nn.functional as F

class NyAttentioin(torch.nn.Module): #简单注意力机制
    def __init__(self, hidden_size, attensize_size):
        super(NyAttentioin, self).__init__()

        self.attn = SelfAttention(hidden_size=hidden_size)
        self.ctx = torch.nn.Linear(in_features=attensize_size, out_features=1, bias=False)

    # inputs: [b, seq_len, hidden_size]
    def forward(self, inputs):
        u = self.attn(inputs) # [b, seq_len, hidden_size]=>[b, seq_len, attention_size]
        scores = self.ctx(u) # [b, seq_len, attention_size]=>[b, seq_len, 1]
        attn_weights = F.softmax(scores, dim=1) # [b, seq_len, 1]

        out = torch.bmm(inputs.transpose(1, 2), attn_weights) # [b, seq_len, hidden_size]=>[b, hidden_size, seq_len]x[b, seq_len, 1]=>[b, hidden_size, 1]
        return torch.squeeze(out, dim=-1) # [b, hidden_size, 1]=>[b, hidden_size]


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.W_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, inputs):
        # Linear transformations to obtain query, key, and value
        Q = self.W_q(inputs)
        K = self.W_k(inputs)
        V = self.W_v(inputs)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(inputs.size(-1)).float())

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)

        return output

if __name__ == '__main__':
    input = torch.rand((2,2048,768))
    att = NyAttentioin(hidden_size=768,attensize_size=768)
    output = att(input)
    print(output)