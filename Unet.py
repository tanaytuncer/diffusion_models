"""
Probabilistic Diffusion Model U-Net architecture

"""

import math

import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class sigmoid_function(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):

    def __init__(self, n_dim):

        super().__init__()
        self.n_dim = n_dim
        self.fc1 = nn.Linear(self.n_dim // 4, self.n_dim)
        self.act = sigmoid_function()
        self.fc2 = nn.Linear(self.n_dim, self.n_dim)

        self.apply(init_weights)

    def forward(self, t):
        half_dim = self.n_dim // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.fc1(emb))
        emb = self.fc2(emb)
        return emb


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, time_channels, dropout = 0.1):

        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_dim)
        self.act1 = sigmoid_function()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = nn.GroupNorm(32, out_dim)
        self.act2 = sigmoid_function()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), padding=(1, 1))

        self.time_emb = nn.Linear(time_channels, out_dim)
        self.time_act = sigmoid_function()

        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.apply(init_weights)

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):

    def __init__(self, n_dim, n_heads = 1, d_k = None):

        super().__init__()

        if d_k is None:
            d_k = n_dim

        self.norm = nn.GroupNorm(32, n_dim)
        self.projection = nn.Linear(n_dim, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_dim)
        self.scaling = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

        self.apply(init_weights)

    def forward(self, x, t = None):
        _ = t
        n_batch, n_dim, h, w = x.shape

        x = x.view(n_batch, n_dim, -1).permute(0, 2, 1)

        qkv = self.projection(x).view(n_batch, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scaling
        attn = attn.softmax(dim=2)

        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(n_batch, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x
        res = res.permute(0, 2, 1).view(n_batch, n_dim, h, w)
        return res


class DownBlock(nn.Module):

    def __init__(self, in_dim, out_dim, time_channels, attn_mechnsm):
        super().__init__()
        self.res = ResidualBlock(in_dim, out_dim, time_channels)

        if attn_mechnsm:
            self.attn = AttentionBlock(out_dim)
        else:
            self.attn = nn.Identity()

        self.apply(init_weights)

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim, time_channels, attn_mechnsm):
        super().__init__()

        self.res = ResidualBlock(in_dim + out_dim, out_dim, time_channels)
        if attn_mechnsm:
            self.attn = AttentionBlock(out_dim)
        else:
            self.attn = nn.Identity()

        self.apply(init_weights)

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):

    def __init__(self, n_dim, time_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_dim, n_dim, time_channels)
        self.attn = AttentionBlock(n_dim)
        self.res2 = ResidualBlock(n_dim, n_dim, time_channels)

        self.apply(init_weights)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):

    def __init__(self, n_dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_dim, n_dim, (4, 4), (2, 2), (1, 1))

        self.apply(init_weights)

    def forward(self, x, t):
        _ = t
        x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, n_dim):
        super().__init__()
        self.conv = nn.Conv2d(n_dim, n_dim, (3, 3), (2, 2), (1, 1))

        self.apply(init_weights)

    def forward(self, x, t):
        _ = t
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, img_dim = 3, n_dim = 32, ch_mults = (1, 2, 2, 4), attn_mechnsm = (False, False, True, True), n_blocks = 2):

        super().__init__()
        n_resolutions = len(ch_mults)
        self.image_projection = nn.Conv2d(img_dim, n_dim, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(n_dim * 4)

        down_step_n = []
        out_dim = in_dim = n_dim
        for i in range(n_resolutions):
            out_dim = in_dim * ch_mults[i]
            for _ in range(n_blocks):
                down_step_n.append(DownBlock(in_dim, out_dim, n_dim * 4, attn_mechnsm[i]))
                in_dim = out_dim
            if i < n_resolutions - 1:
                down_step_n.append(Downsample(in_dim))

        self.down_smpl_steps = nn.ModuleList(down_step_n)

        self.middle_smpl_step = MiddleBlock(out_dim, n_dim * 4, )

        up_step_n = []
        in_dim = out_dim
        for i in reversed(range(n_resolutions)):
            out_dim = in_dim
            for _ in range(n_blocks):
                up_step_n.append(UpBlock(in_dim, out_dim, n_dim * 4, attn_mechnsm[i]))
            out_dim = in_dim // ch_mults[i]
            up_step_n.append(UpBlock(in_dim, out_dim, n_dim * 4, attn_mechnsm[i]))
            in_dim = out_dim
            if i > 0:
                up_step_n.append(Upsample(in_dim))

        self.up_smpl_steps = nn.ModuleList(up_step_n)

        self.norm = nn.GroupNorm(8, n_dim)
        self.act = sigmoid_function()
        self.resl = nn.Conv2d(in_dim, img_dim, kernel_size=(3, 3), padding=(1, 1))

        self.apply(init_weights)

    def forward(self, x, t):

        t = self.time_emb(t)
        x = self.image_projection(x)
        h = [x]

        for m in self.down_smpl_steps:
            x = m(x, t)
            h.append(x)

        x = self.middle_smpl_step(x, t)

        for m in self.up_smpl_steps:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.resl(self.act(self.norm(x)))