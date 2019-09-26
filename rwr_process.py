import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
np.set_printoptions(threshold=np.nan)

class RWRLayer(nn.Module):
    """
    adaptive structural fingerprint layer
    """

    def __init__(self, in_features, out_features, dropout, alpha,adj_ad, concat=True,):
        super(RWRLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj_ad=adj_ad
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N* N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        s=self.adj_ad
        fw = open('dijskra_citeseer.pkl', 'rb')
        dijkstra = pickle.load(fw)
        Dijkstra = dijkstra.numpy()
        ri_all = []
        ri_index = []
        # You may replace 3327 with the size of dataset
        for i in range(3327):
            # You may replace 1,4 with the .n-hop neighbors you want
            index_i = np.where((Dijkstra[i] < 4) & (Dijkstra[i] > 1))
            I = np.eye((len(index_i[0]) + 1), dtype=int)
            ei = []
            for q in range((len(index_i[0]) + 1)):
                if q == 0:
                    ei.append([1])
                else:
                    ei.append([0])
            W = []
            for j in range((len(index_i[0])) + 1):
                w = []
                for k in range((len(index_i[0])) + 1):
                    if j == 0:
                        if k == 0:
                            w.append(float(0))
                        else:
                            w.append(float(1))
                    else:
                        if k == 0:
                            w.append(float(1))
                        else:
                            w.append(float(0))
                W.append(w)
            # the choice of the c parameter in RWR
            c = 0.5
            W = np.array(W)
            rw_left = (I - c * W)
            try:
                rw_left = np.linalg.inv(rw_left)
            except:
                rw_left = rw_left
            else:
                rw_left = rw_left
            ei = np.array(ei)
            rw_left = torch.tensor(rw_left, dtype=torch.float32)
            ei = torch.tensor(ei, dtype=torch.float32)
            ri = torch.mm(rw_left, ei)
            ri = torch.transpose(ri, 1, 0)
            ri = abs(ri[0]).numpy().tolist()
            ri_index.append(index_i[0])
            ri_all.append(ri)

        fw = open('ri_index_c_0.5_citeseer_highorder_1_x_abs.pkl', 'wb')
        pickle.dump(ri_index, fw)
        fw.close()

        fw = open('ri_all_c_0.5_citeseer_highorder_1_x_abs.pkl', 'wb')
        pickle.dump(ri_all, fw)
        fw.close()

        e = e.cuda()
        zero_vec = -9e15*torch.ones_like(e)
        k_vec=-9e15*torch.ones_like(e)
        adj=adj.cuda()
        np.set_printoptions(threshold=np.nan)
        attention = torch.where(adj>0 , e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




