import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self._out_ch = out_ch
        self._ds_ch = 1024

        self.inc = DoubleConv(in_ch, 64)

        self.down = nn.ModuleList()
        for m in [           # in: 256x
            DownConv(64, 128),   # 128x
            DownConv(128, 256),  # 64x
            DownConv(256, 512),  # 32x
            DownConv(512, 1024)  # 16x
        ]:
            self.down.append(m)

        self.up = nn.ModuleList()
        for m in [
            UpConv(1024, 512), # 32x
            UpConv(512, 256),  # 64x
            UpConv(256, 128),  # 128x
            UpConv(128, 64)    # 256x
        ]:
            self.up.append(m)

        self.outc = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    
    @property
    def out_ch(self):
        return self._out_ch

    @property
    def ds_ch(self):
        return self._ds_ch

    def forward(self, x):
        # x: [B, M, C, W, H], M: the number of views
        # outputs: [B, M, C', W', H']
        b, m = x.shape[:2]
        x = x.reshape(b * m, *x.shape[2:])

        x1 = self.inc(x)

        xs = [x1]
        for conv in self.down:
            xs.append(conv(xs[-1]))

        x = xs[-1]
        ds_x = x

        for i, conv in enumerate(self.up):
            x = conv(x, xs[-(2 + i)])

        y = self.outc(x)
        return {
            'feats': y.reshape(b, m, *y.shape[1:]), 
            'feats_ds': ds_x.reshape(b, m, *ds_x.shape[1:])
        }


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        '''
           if you have padding issues, see
           https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
           https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
