import torch as th
from torch import nn

class MIND(nn.Module):
    def __init__(self, args, embedNum):
        super(MIND, self).__init__()
        self.D = args.D
        self.K = args.K
        self.R = args.R
        self.L = args.seq_len
        self.nNeg = args.n_neg
        # 权重初始化
        self.itemEmbeds = th.nn.Embedding(embedNum, self.D, padding_idx=0)
        self.dense1 = th.nn.Linear(self.D, 4 * self.D)
        self.dense2 = th.nn.Linear(4 * self.D, self.D)
        # 一个S用于所有路由操作，第一个dim用于批量广播
        S = th.empty(self.D, self.D)
        th.nn.init.normal_(S, mean=0.0, std=1.0)
        self.S = th.nn.Parameter(S)
        # 初始化后固定路由日志
        self.B = th.nn.init.normal_(th.empty(self.K, self.L), mean=0.0, std=1.0)
        self.opt = th.optim.Adam(self.parameters(), lr=args.lr)

    # 胶囊神经网络中的激活函数，用于对向量进行非线性压缩，将输入向量的模值压缩到 0 到 1 之间，并保持向量的方向信息
    def squash(self, caps, bs):
        n = th.norm(caps, dim=2).view(bs, self.K, 1)
        nSquare = th.pow(n, 2)
        return (nSquare / ((1 + nSquare) * n + 1e-9)) * caps
    
    def B2IRouting(self, his, bs):
        # B2I 动态路由、输入行为、输出上限
        B = self.B.detach()
        # 除了第一轮路由外，每个样本的 w 都是不同的，因此需要一个 dim 来进行批量处理
        B = th.tile(B, (bs, 1, 1)) # (bs, K, L)
        # masking，使padding索引的路由logit为INT_MAX，以便softmax结果为0
        # (bs, L) -> (bs, 1, L) -> (bs, K, L)
        mask = (his != 0).unsqueeze(1).tile(1, self.K, 1)
        drop = (th.ones_like(mask) * -(1 << 31)).type(th.float32)
        # 将历史行为嵌入
        his = self.itemEmbeds(his)
        his = th.matmul(his, self.S)

        for i in range(self.R):
            BMasked = th.where(mask, B, drop)
            W = th.softmax(BMasked, dim=2)
            if i < self.R - 1:
                with th.no_grad():
                    # 所有 i 到每个 j 的加权和
                    caps = th.matmul(W, his)
                    caps = self.squash(caps, bs)
                    B += th.matmul(caps, th.transpose(his, 1, 2))
            else:
                caps = th.matmul(W, his)
                caps = self.squash(caps, bs)
                # 跳过上一轮的路由logits更新

        caps = self.dense2(th.relu(self.dense1(caps)))
        #caps = caps / (th.norm(caps, dim=2).view(bs, self.K, 1) + 1e-9)
        
        return caps
    
    def labelAwareAttation(self, caps, tar, p=2):
        """ 标签感知注意力、输入上限和目标以及输出逻辑
            caps: (bs, K, D)
            tar: (bs, cnt, D)
            对于正例, cnt = 1
            对于负例, cnt = self.nNeg
        """
        tar = tar.transpose(1, 2)
        w = th.softmax(
                # (bs, K, D) X (bs, D, cnt) -> (bs, K, cnt) -> (bs, cnt, K)
                th.pow(th.transpose(th.matmul(caps, tar), 1, 2), p),
                dim=2
            )
        w = w.unsqueeze(2)

        # (bs, cnt, 1, K) X (bs, 1, K, D) -> (bs, cnt, 1, D) -> (bs, cnt, D)
        caps = th.matmul(w, caps.unsqueeze(1)).squeeze(2)
        return caps

    def sampledSoftmax(self, caps, tar, bs, tmp=0.01):
        tarPos = self.itemEmbeds(tar) # (bs, D)
        capsPos = self.labelAwareAttation(caps, tarPos.unsqueeze(1)).squeeze(1)
        #his = his / (th.norm(his, dim=1).view(bs, 1) + 1e-9)
        #tar = tar / (th.norm(tar, dim=1).view(bs, 1) + 1e-9)
        # (bs, D) dot (bs, D) -> (bs, D) - sum > (bs, )
        posLogits = th.sigmoid(th.sum(capsPos * tarPos, dim=1) / tmp)

        # neg logits
        # 批量负采样
        tarNeg = tarPos[th.multinomial(th.ones(bs), self.nNeg * bs, replacement=True)].view(bs, self.nNeg, self.D) # (batch_size, nNeg, D)
        capsNeg = self.labelAwareAttation(caps, tarNeg)
        # (bs, nNeg, dim) -> (bs, nNeg, 1) -> (bs * nNeg, )
        negLogits = th.sigmoid(th.sum(capsNeg * tarNeg, dim=2).view(bs * self.nNeg) / tmp)

        logits = th.concat([posLogits, negLogits])
        labels = th.concat([th.ones(bs, ), th.zeros(bs * self.nNeg)])

        return logits, labels