"""
YOLOv8-Lite 轻量化模块
"""
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv as OfficialConv

try:
    from ultralytics.nn.modules.block import C2f as OfficialC2f
except Exception as _e:  # pragma: no cover
    raise ImportError(
        "无法从 ultralytics.nn.modules.block 导入官方 C2f"
    ) from _e


# ----------------------------- Ghost 系列内部实现 ----------------------------- #

class _GhostConvInner(nn.Module):
    """Ghost Convolution：少量标准卷积生成本征特征图 + 廉价深度可分离卷积生成"重影"特征图"""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        c_ = max(c2 // 2, 1)
        self.cv1 = OfficialConv(c1, c_, k, s, act=act)
        self.cv2 = OfficialConv(c_, c_, 5, 1, g=c_, act=act)  # depthwise，生成"重影"特征

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class _GhostBottleneckInner(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = max(int(c2 * e), 1)
        self.cv1 = _GhostConvInner(c1, c_, 1, 1)
        self.cv2 = _GhostConvInner(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class _GhostC2fInner(nn.Module):
    """结构与官方 C2f 完全一致（cv1 分流 + n 个瓶颈块 + cv2 融合），只是内部瓶颈块换成 Ghost 版本"""

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = max(int(c2 * e), 1)
        self.cv1 = OfficialConv(c1, 2 * self.c, 1, 1)
        self.cv2 = OfficialConv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(_GhostBottleneckInner(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class _GSConvInner(nn.Module):
    """Group-Shuffle Conv：标准卷积(一半通道) + 深度可分离卷积 + Channel Shuffle"""

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        c_ = max(c2 // 2, 1)
        self.cv1 = OfficialConv(c1, c_, k, s, act=act)
        self.cv2 = OfficialConv(c_, c_, 5, 1, g=c_, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.shape
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w).permute(1, 0, 2).reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


# ----------------------------- 对外顶替官方名字的包装类 ----------------------------- #

class LiteC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, ghost=False):
        super().__init__()
        if ghost:
            self.impl = _GhostC2fInner(c1, c2, n, shortcut, e)
        else:
            self.impl = OfficialC2f(c1, c2, n, shortcut, g, e)

    def forward(self, x):
        return self.impl(x)

    def forward_split(self, x):
        # 兼容官方 C2f 可能被调用 forward_split（chunk 拆分写法）的情况
        if hasattr(self.impl, "forward_split"):
            return self.impl.forward_split(x)
        return self.forward(x)


class LiteConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, slim=False):
        super().__init__()
        if slim:
            self.impl = _GSConvInner(c1, c2, k, s, act)
        else:
            self.impl = OfficialConv(c1, c2, k, s, p, g, d, act)

    def forward(self, x):
        return self.impl(x)

    def forward_fuse(self, x):
        if hasattr(self.impl, "forward_fuse"):
            return self.impl.forward_fuse(x)
        return self.forward(x)
