import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, T, dim_in, dim_out):
        super().__init__()
        assert dim_in % 2 == 0
        emb = torch.arange(0, dim_in, step=2) / dim_in * math.log(10000)  # [0, 2, 4, ..., dim_in//2]
        emb = torch.exp(emb)  # shape = [dim_in//2]
        pos = torch.arange(T).float()  # shape = [1000]
        emb = pos[:, None] * emb[None, :]  #  [1000, 1] * [1, dim_in//2] = [1000, dim_in//2]
        assert list(emb.shape) == [T, dim_in // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1) # [1000, dim_in//2] stack [1000, dim_in//2] = [1000, dim_in//2, 2]
        assert list(emb.shape) == [T, dim_in // 2, 2]
        emb = emb.view(T, dim_in)  # [1000, dim_in//2, 2] -> [1000, dim_in]
        
        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # 以上述emb为值生成embedding
            nn.Linear(dim_in, dim_out),
            Swish(),
            nn.Linear(dim_out, dim_out)
        )
        
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, t):
        emb = self.time_embedding(t)
        return emb
                

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, t_dim, dropout, attn=False):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_dim),  # 分32组
            Swish(),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )
        
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(t_dim, out_dim)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )
        
        if in_dim != out_dim:
            self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        
        if attn:
            self.attn = AttnBlock(out_dim)
        else:
            self.attn = nn.Identity()
        
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
    
    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    
    
class AttnBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_dim)
        self.proj_q = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        
        self.initialize()
        
    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        
        q = q.permute(0, 2, 3, 1).view(B, H * W, C) # [B, C, H, W] -> [B, H, W, C] ->[B, seq_len, C]
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)
        
        v = v.permute(0, 2, 3, 1).view(B, H * W, C) # [B, C, H, W] -> [B, H, W, C] ->[B, seq_len, C]
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        
        return x + h
    
    
class DownSample(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.downsample = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1)
        self.initialize()
        
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.downsample.weight)
        torch.nn.init.zeros_(self.downsample.bias)
        
    def forward(self, x, t):
        x = self.downsample(x)
        return x
    
class UpSample(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.upsample = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.initialize()
        
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.upsample.weight)
        torch.nn.init.zeros_(self.upsample.bias)
        
    def forward(self, x, t):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upsample(x)
        return x
        

class UNet(nn.Module):
    def __init__(self, T, dim, dim_scale, attn, num_res_blocks, dropout):
        """
        T: int 扩散总步数
        dim: int 初始通道数
        dim_scale: [int, int, ...] 通道数扩张的倍数
        attn: [int, int, ...] 哪些索引的残差块需要添加attn
        num_res_blocks : int 残差块的数量
        dropout: bool 是否使用dropout
        """
        super().__init__()
        
        temb_dim = dim * 4
        self.time_embedding = TimeEmbedding(T, dim, temb_dim)
        
        self.head = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1) # 3 -> dim
        self.downblocks = nn.ModuleList()
        
        dim_list = [dim] # 记录每个block输出特征的通道数
        curr_dim = dim
        
        for idx, scale in enumerate(dim_scale):  # 扩充通道数
            out_dim = dim * scale
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_dim=curr_dim, out_dim=out_dim, t_dim=temb_dim, dropout=dropout, attn=(idx in attn)))
                curr_dim = out_dim
                dim_list.append(curr_dim)
            if idx != len(dim_scale)-1:  # 如果不是最后一个残差块，就要进行下采样
                self.downblocks.append(DownSample(curr_dim))
                dim_list.append(curr_dim)
        
        self.middleblocks = nn.ModuleList([
            ResBlock(curr_dim, curr_dim, temb_dim, dropout, attn=True),
            ResBlock(curr_dim, curr_dim, temb_dim, dropout, attn=False)
        ])

        
        self.upblocks = nn.ModuleList()
        for idx, scale in reversed(list(enumerate(dim_scale))):
            out_dim = dim * scale
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_dim=dim_list.pop()+curr_dim, out_dim=out_dim, t_dim=temb_dim, dropout=dropout, attn=(idx in attn)))
                curr_dim = out_dim
                
            if idx != 0:
                self.upblocks.append(UpSample(curr_dim))
        
        assert len(dim_list) == 0
        
        self.tail = nn.Sequential(
            nn.GroupNorm(32, curr_dim),
            Swish(),
            nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1)
        )

        self.initialize()
        
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.xavier_uniform_(self.tail[-1].weight)
        torch.nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        temb = self.time_embedding(t)

        h = self.head(x)

        h_list = [h]

        for layer in self.downblocks:
            h = layer(h, temb)
            h_list.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb)
        
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, h_list.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail(h)

        assert len(h_list) == 0
        return h

        
if __name__ == "__main__":
    batch_size = 1
    model = UNet(
        T=1000, dim=128, dim_scale=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)
    
        
        