import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class CBOWModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(CBOWModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(2*self.emb_size-1, self.emb_dimension, sparse=True)  # 定义输入词的嵌入字典形式
        self.w_embeddings = nn.Embedding(2*self.emb_size-1, self.emb_dimension, sparse=True)  # 定义输出词的嵌入字典形式
        self._init_embedding()  # 初始化

    def _init_embedding(self):
        int_range = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        self.w_embeddings.weight.data.uniform_(-0, 0)

    def compute_context_matrix(self, u):
        pos_u_emb = []  # 上下文embedding
        for per_Xw in u:
            # 上下文矩阵的第一维不同词值不同，如第一个词上下文为c，第二个词上下文为c+1，需要统一化
            per_u_emb = self.u_embeddings(torch.LongTensor(per_Xw))  # 对上下文每个词转embedding
            per_u_numpy = per_u_emb.data.numpy()  # 转回numpy，好对其求和
            per_u_numpy = np.sum(per_u_numpy, axis=0)
            per_u_list = per_u_numpy.tolist()  # 为上下文词向量Xw的值
            pos_u_emb.append(per_u_list)  # 放回数组
        pos_u_emb = torch.FloatTensor(pos_u_emb)  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        return pos_u_emb

    def forward(self, pos_u, pos_w, neg_u, neg_w, n):
        pos_u_emb = self.compute_context_matrix(pos_u)
        pos_w_emb = self.w_embeddings(torch.LongTensor(pos_w))
        neg_u_emb = self.compute_context_matrix(neg_u)
        neg_w_emb = self.w_embeddings(torch.LongTensor(neg_w))

        # 计算梯度上升（ 结果 *（-1） 即可变为损失函数 ->可使用torch的梯度下降）
        score_1 = torch.mul(pos_u_emb, pos_w_emb).squeeze()  # Xw.T * θu
        score_2 = torch.sum(score_1, dim=1)  # 点积和
        score_3 = F.logsigmoid( - score_2)  # left side log (1 - sigmoid (Xw.T * θw))
        neg_score_1 = torch.mul(neg_u_emb, neg_w_emb).squeeze()  # Xw.T * θw
        neg_score_2 = torch.sum(neg_score_1, dim=1)  # 点积和
        neg_score_3 = F.logsigmoid(neg_score_2)  #right side log sigmoid (Xw.T * θneg(w))
        # L = log sigmoid (Xw.T * θw) + logsigmoid (-Xw.T * θw)
        loss = torch.sum(score_3) + torch.sum(neg_score_3)
        # n: how many words we can considered here
        return -1 * loss / n

    # 存储embedding
    def save_embedding(self, id2word_dict, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        file_output = open(file_name, 'w', encoding="utf8")
        file_output.write('%d %d\n' % (self.emb_size, self.emb_dimension))
        for id, word in id2word_dict.items():
            e = embedding[id]
            e = ' '.join(map(lambda x: str(x), e))
            file_output.write('%s %s\n' % (word, e))


