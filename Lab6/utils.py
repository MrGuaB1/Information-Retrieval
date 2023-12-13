import tqdm
import numpy as np
import torch as th
import argparse


def parseArgs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--min_user_freq', type=int, default=20)
    argparser.add_argument('--min_item_freq', type=int, default=100)
    argparser.add_argument('--train_user_frac', type=float, default=0.8)
    argparser.add_argument('--seq_len', type=int, default=20)
    argparser.add_argument('--train_batch_size', type=int, default=1024)
    argparser.add_argument('--test_batch_size', type=int, default=8)
    argparser.add_argument('--n_neg', type=int, default=10)
    argparser.add_argument('--D', type=int, default=8)
    argparser.add_argument('--K', type=int, default=3)
    argparser.add_argument('--R', type=int, default=3)
    argparser.add_argument('--print_steps', type=int, default=200)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.001)

    return argparser.parse_args()


# 通过滑动窗口生成用户样本
def genUserTrainSamples(args, userDf):
    # 一个用户生成多个训练样本
    userDf.reset_index(drop=True, inplace=True)
    his, tar = [], []
    for i in range(1, userDf.shape[0]):
        # x = window [i - SEQ_LEN, i - 1], y = item[i]
        his.append(padOrCut(userDf.iloc[max(0, i - args.seq_len):i]['itemId'].values, args.seq_len))
        tar.append(userDf.iloc[i]['itemId'])

    return np.stack(his), np.stack(tar)


def genUserTestSamples(args, userDf):
    # one user generate one test sample
    userDf.reset_index(drop=True, inplace=True)
    idx = int(0.8 * userDf.shape[0])
    his = padOrCut(userDf['itemId'].iloc[:idx].values, args.seq_len)
    tar = userDf['itemId'].iloc[idx:].values

    return his, tar


def padOrCut(seq, L):
    if (len(seq) < L): return np.concatenate([seq, (L - len(seq)) * [0]])
    # return last len
    elif (len(seq) > L): return seq[len(seq) - L:]
    else: return seq


class Dataset:
    def __init__(self, his, tar):
        self.his = his
        self.tar = tar
        assert len(self.his) == len(self.tar)
    
    def __getitem__(self, i):
        return self.his[i], self.tar[i]
    
    def __len__(self):
        return len(self.his)

# 测试函数，主要是使用训练好的模型对测试集中的数据进行预测，获取 top-N 推荐结果
def test(model, testData, _testTar, top=30):
    with th.no_grad():
        ie = model.itemEmbeds.weight
        N = ie.shape[0]
        # user-wise recall(0~1) and hit (0 or 1)
        recalls, hitRates = [], []
        # 计算用户行为的表示向量
        for his, tar in tqdm.tqdm(testData):
            bs = his.shape[0]
            caps = model.B2IRouting(his, bs) # (bs, K, D)

            # 将计算得到的表示向量与物品的嵌入向量进行矩阵乘法，得到用户与所有物品之间的推荐评分
            # (bs, K, D) X (bs, D, N) -> (bs, K, N) -> (bs, K * N)
            logits = th.matmul(caps, th.transpose(ie, 0, 1)).view(bs, model.K * N).detach().numpy()
            # quick select over dim 1
            res = np.argpartition(logits, kth=N - top, axis=1)[:, -top:] # (bs, top)
            hits = 0
            for r, t in zip(res, tar):
                t = [x for x in _testTar[t] if x != 0]
                if not t: continue
                r = set(r)
                # 对于每个用户，将推荐的物品与实际测试集中的物品进行比较，计算召回率和命中率
                for i in t:
                    if (i in r): hits += 1
                recalls.append(hits / len(t))
                hitRates.append(1 if hits > 0 else 0)

        print(f"recall@{top}: {np.mean(recalls)}, hitRate@{top}: {np.mean(hitRates)}")