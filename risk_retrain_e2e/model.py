from __future__ import annotations
import torch
from torch import nn

LABELS=("NO_EVIDENCE","REVIEW","HIGH_EVIDENCE")

class EndToEndRiskModel(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__(); self.encoder=encoder
        width=int(encoder.hidden_size)*2
        self.dropout=nn.Dropout(0.15)
        self.classifier=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),nn.Dropout(0.15),nn.Linear(256,len(LABELS)))
    def forward(self,batch:dict[str,torch.Tensor])->torch.Tensor:
        encoded=self.encoder(batch)
        rep=torch.cat((encoded.prefix,encoded.current),dim=-1)
        return self.classifier(self.dropout(rep))
