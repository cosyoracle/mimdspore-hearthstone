import mindspore.nn as nn
import mindspore
import math

from mindspore import Parameter


class Attention(nn.Cell):

    # 注意力机制
    def construct(self, query, key, value, mask=None, dropout=None):
        scores = mindspore.ops.matmul(query, key.swapaxes(-2, -1)) \
                 / math.sqrt(query.shape[-1])
        if mask is not None:
            if len(list(mask.shape)) != 4:
                mask = mask.unsqueeze(1).tile((1, query.shape[2], 1)).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = mindspore.ops.softmax(scores, axis=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return mindspore.ops.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Cell):

    # 考虑模型尺寸和头部数量
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.CellList([nn.Dense(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Dense(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # 1) 批量进行所有线性投影 => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).swapaxes(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # 2) 将注意力集中在批量中的所有投影向量上
        x, attn = self.attention(query, key, value, mask=mask, dropout=None)
        # 3) "Concat" 使用视图并应用最终的线性
        x = x.swapaxes(1, 2).view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class GELU(nn.Cell):

    def construct(self, x):
        return 0.5 * x * (1 + mindspore.ops.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * mindspore.ops.pow(x, 3))))


class DenseLayer(nn.Cell):

    # 实现FFN方程
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(DenseLayer, self).__init__()
        self.w_1 = nn.Dense(d_model, d_ff)
        self.w_2 = nn.Dense(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def construct(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Cell):

    # 层次模块
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = mindspore.Parameter(mindspore.ops.ones(features))
        self.b_2 = mindspore.Parameter(mindspore.ops.zeros(features))
        self.eps = eps

    def construct(self, x):
        mean = x.mean(-1, keep_dims=True)
        std = x.std(-1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class SublayerConnection(nn.Cell):

    # 残差连接
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def construct(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class CombinationLayer(nn.Cell):

    def construct(self, query, key, value, dropout=None):
        query_key = query * key / math.sqrt(query.shape[-1])
        query_value = query * value / math.sqrt(query.shape[-1])
        tmpW = mindspore.ops.stack([query_key, query_value], -1)
        tmpsum = mindspore.ops.softmax(tmpW, axis=-1)
        tmpV = mindspore.ops.stack([key, value], axis=-1)
        tmpsum = tmpsum * tmpV
        tmpsum = mindspore.ops.squeeze(mindspore.ops.sum(tmpsum, dim=-1), -1)
        if dropout:
            tmpsum = dropout(tmpsum)
        return tmpsum


class MultiHeadedCombination(nn.Cell):

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.CellList([nn.Dense(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Dense(d_model, d_model)
        self.combination = CombinationLayer()
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, query, key, value, mask=None, batch_size=-1):
        if batch_size == -1:
            batch_size = query.shape[0]
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).swapaxes(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x = self.combination(query, key, value, dropout=self.dropout)

        x = x.swapaxes(1, 2).view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class GCNN(nn.Cell):

    def __init__(self, dmodel):
        super(GCNN, self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Dense(dmodel, dmodel)
        self.linearSecond = nn.Dense(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.com = MultiHeadedCombination(8, dmodel)

    def construct(self, state, left, inputad):
        state = mindspore.ops.cat([left, state], axis=1)
        state = self.linear(state)
        degree = mindspore.ops.sum(inputad, dim=-1, keepdim=True).clamp(min=1e-6)
        degree2 = mindspore.ops.sum(inputad, dim=-2, keepdim=True).clamp(min=1e-6)

        degree = 1.0 / mindspore.ops.sqrt(degree)
        degree2 = 1.0 / mindspore.ops.sqrt(degree2)
        degree2 = degree2 * inputad * degree
        state = self.subconnect(state, lambda _x: self.com(_x, _x, mindspore.ops.matmul(degree2, state)))
        state = self.linearSecond(state)
        return state[:, 50:, :]


class GCNNM(nn.Cell):

    def __init__(self, dmodel):
        super(GCNNM, self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Dense(dmodel, dmodel)
        self.linearSecond = nn.Dense(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.com = MultiHeadedCombination(8, dmodel)
        self.comb = MultiHeadedCombination(8, dmodel)
        self.subconnect1 = SublayerConnection(dmodel, 0.1)

    def construct(self, state, inputad, rule):
        state = self.subconnect1(state, lambda _x: self.comb(_x, _x, rule, batch_size=1))
        state = self.linear(state)
        degree = mindspore.ops.sum(inputad, dim=-1, keepdim=True).clamp(min=1e-6)
        degree2 = mindspore.ops.sum(inputad, dim=-2, keepdim=True).clamp(min=1e-6)
        degree = 1.0 / mindspore.ops.sqrt(degree)
        degree2 = 1.0 / mindspore.ops.sqrt(degree2)
        degree2 = degree2 * inputad * degree
        state2 = mindspore.ops.matmul(degree2, state)
        state = self.subconnect(state, lambda _x: self.com(_x, _x, state2, batch_size=1))
        return state


class ConvolutionLayer(nn.Cell):
    def __init__(self, dmodel, layernum, kernelsize=3, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.conv1 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize-1)//2, pad_mode='pad')
        self.conv2 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize-1)//2, pad_mode='pad')
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def construct(self, x, mask):
        convx = self.conv1(x.permute(0, 2, 1))
        convx = self.conv2(convx)
        out = self.dropout(self.activation(convx.permute(0, 2, 1)))
        return out


class PositionalEmbedding(nn.Cell):

    def __init__(self, d_model, max_len=1024):
        super().__init__()

        pe = mindspore.ops.zeros((max_len, d_model)).float()
        pe.require_grad = False
        position = mindspore.ops.arange(0, max_len).float().unsqueeze(1)
        div_term = (mindspore.ops.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = mindspore.ops.sin(position * div_term)
        pe[:, 1::2] = mindspore.ops.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = Parameter(pe, name='pe', requires_grad=False)

    def construct(self, x):
        return self.pe[:, :x.shape[-1]]


class rightTransformerBlock(nn.Cell):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention1 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention2 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.combination = MultiHeadedCombination(h=attn_heads, d_model=hidden)
        self.feed_forward = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.conv_forward = ConvolutionLayer(dmodel=hidden, layernum=hidden)
        self.Tconv_forward = GCNN(dmodel=hidden)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, mask, inputleft, leftmask, charEm, inputP):
        x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, charEm))
        x = self.sublayer3(x, lambda _x: self.attention2.forward(_x, inputleft, inputleft, mask=leftmask))
        x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, inputleft, inputP))
        return self.dropout(x)


class TransformerBlock(nn.Cell):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.combination = MultiHeadedCombination(h=attn_heads, d_model=hidden)
        self.feed_forward = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.conv_forward = ConvolutionLayer(dmodel=hidden, layernum=hidden)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, mask, charEm, treemask=None, isTree=False):
        x = self.sublayer1(x, lambda _x: self.attention.construct(_x, _x, _x, mask=mask))
        x = self.sublayer2(x, lambda _x: self.combination.construct(_x, _x, charEm))
        if isTree:
            x = self.sublayer3(x, lambda _x: self.attention.construct(_x, _x, _x, mask=treemask))
            x = self.sublayer4(x, self.feed_forward)
        else:
            x = self.sublayer3(x, lambda _x: self.conv_forward.construct(_x, mask))
        return self.dropout(x)


class Embedding(nn.Cell):

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_size)
        self.depth_embedding = nn.Embedding(20, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def construct(self, sequence, inputdept=None, usedepth=False):
        x = self.token(sequence) + self.position(sequence)
        if usedepth:
            x = x + self.depth_embedding(inputdept)
        return self.dropout(x)


class decodeTransformerBlock(nn.Cell):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):

        super().__init__()
        self.attention1 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention2 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, mask, inputleft, leftmask, inputleft2, leftmask2):
        x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, inputleft, inputleft, mask=leftmask))
        x = self.sublayer3(x, lambda _x: self.attention2.forward(_x, inputleft2, inputleft2, mask=leftmask2))
        x = self.sublayer4(x, self.feed_forward)
        return self.dropout(x)



