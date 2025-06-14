import torch

from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 入力の28×28画像を1次元ベクトルに変換 (784次元)

        self.activation = nn.Sequential(
            nn.Linear(28*28, 512), # 入力784 → 隠れ層512
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # 出力層（10クラス分類）
        )
    def forward(self,x):
        
        x = self.flatten(x)
        
        logits = self.activation(x)

        return logits
    
