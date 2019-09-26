import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import StructuralFingerprintLayer
from rwr_process import RWRLayer

class ADSF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad):
        """version of ADSF."""
        super(ADSF, self).__init__()
        self.dropout = dropout
        self.attentions = [StructuralFingerprintLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad,concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att =StructuralFingerprintLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,adj_ad=adj_ad, concat=False)

    def forward(self, x, adj,adj_ad):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)




class RWR_process(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad):
        """version of RWR_process."""
        super(RWR_process, self).__init__()
        self.dropout = dropout
        self.attentions = [
            RWRLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, concat=True) for _ in
            range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = RWRLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adj_ad=adj_ad,
                                                  concat=False)

    def forward(self, x, adj, adj_ad):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

