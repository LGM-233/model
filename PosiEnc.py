import torch
class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_num_seg, d_model, num_seg, seg_len):
        super(PositionalEncoding, self).__init__()

        
        if seg_len == 512:
            position = torch.arange(seg_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model))
            pos_enc = torch.zeros((seg_len, d_model),dtype=torch.float32)
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
            pos_enc = pos_enc.unsqueeze(0)  
            self.register_buffer('pos_enc', pos_enc)
        elif seg_len < 512:
            position = torch.arange(0, max_num_seg, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model))
            pos_enc = torch.zeros((max_num_seg, d_model), dtype=torch.float32)
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
            pos_enc = pos_enc[:num_seg]
            self.register_buffer('pos_enc', pos_enc)
        else:
            position = torch.arange(seg_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model))
            pos_enc = torch.zeros((seg_len, d_model), dtype=torch.float32)
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
            pos_enc = pos_enc.unsqueeze(0) 
            self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        
        x = x + self.pos_enc
        return x
