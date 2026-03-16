import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

# 对应论文中的 Stage 1: CSMM (Channel and Space Mixing Module)
def CSMM(c1, c2, depth, kernel_size=3, patch_size=1):


    csmm = nn.Sequential(
        # Patch Embedding
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=1, padding=patch_size//2),
        nn.GELU(), 
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            # Depthwise Convolution 
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=c2),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            )),
            # Pointwise Convolution
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.GELU(),
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return csmm


class SEAM(nn.Module):
    def __init__(self, c1, c2, n=1, reduction=16):
        super(SEAM, self).__init__()
        if c1 != c2:
            c2 = c1
            
       
        self.csmm_p6 = CSMM(c2, c2, depth=n, patch_size=6)
        self.csmm_p7 = CSMM(c2, c2, depth=n, patch_size=7)
        self.csmm_p8 = CSMM(c2, c2, depth=n, patch_size=8)
        
    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
      
        self.mlp = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid() 
        )

        self._initialize_weights()

    def forward(self, x):
        b, c, _, _ = x.size()
        
 
        x_multi = x + self.csmm_p6(x) + self.csmm_p7(x) + self.csmm_p8(x)
        
       
        z = self.avg_pool(x_multi).view(b, c)
        z = self.mlp(z).view(b, c, 1, 1)
        
        
        w_exp = torch.exp(z)
        
       
        return x * w_exp.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
