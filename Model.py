import mindspore.nn as nn
import mindspore
from Utils import TransformerBlock
from Utils import rightTransformerBlock
from Utils import Embedding
from Utils import GELU
from Utils import LayerNorm
from Utils import decodeTransformerBlock
from Utils import GCNNM
from Utils import PositionalEmbedding


class AttEncoder(nn.Cell):

    def __init__(self, args):
        super(AttEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.CellList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])
        self.transformerBlocksTree = nn.CellList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])

    def construct(self, input_code, input_codechar, inputAd):
        codemask = mindspore.ops.gt(input_code, 0)
        charEm = self.char_embedding(input_codechar)
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(axis=-2)
        x = self.token_embedding(input_code.long())
        for trans in self.transformerBlocksTree:
            x = trans.construct(x, codemask, charEm, inputAd, True)
        for trans in self.transformerBlocks:
            x = trans.construct(x, codemask, charEm)
        return x


class NlEncoder(nn.Cell):

    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.CellList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.token_embedding = Embedding(args.Nl_Vocsize, self.embedding_size)

    def construct(self, input_nl, input_nlchar):

        nlmask = mindspore.ops.gt(input_nl, 0)
        charEm = self.char_embedding(input_nlchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(axis=-2)
        x = self.token_embedding(input_nl.long())
        for trans in self.transformerBlocks:
            x = trans.construct(x, nlmask, charEm)
        return x, nlmask


class CopyNet(nn.Cell):

    def __init__(self, args):
        super(CopyNet, self).__init__()
        self.embedding_size = args.embedding_size
        self.LinearSource = nn.Dense(self.embedding_size, self.embedding_size, has_bias=False)
        self.LinearTarget = nn.Dense(self.embedding_size, self.embedding_size, has_bias=False)
        self.LinearRes = nn.Dense(self.embedding_size, 1)
        self.LinearProb = nn.Dense(self.embedding_size, 2)

    def construct(self, source, traget):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(traget)
        genP = self.LinearRes(mindspore.ops.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(
            axis=-1)
        prob = mindspore.ops.softmax(self.LinearProb(traget), axis=-1)
        return genP, prob


class Decoder(nn.Cell):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.word_len = args.WoLen
        self.nl_len = args.NlLen
        self.code_len = args.CodeLen
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, args.WoLen))
        self.path_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.rule_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 2))
        self.depth_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 40))
        self.resLen = args.rulenum - args.NlLen
        self.encodeTransformerBlock = nn.CellList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(9)])
        self.decodeTransformerBlocksP = nn.CellList(
            [decodeTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(2)])
        self.finalLinear = nn.CellList(self.embedding_size, 2048)
        self.resLinear = nn.CellList(2048, self.resLen)
        self.rule_token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        self.rule_embedding = nn.Embedding(args.rulenum, self.embedding_size)
        self.encoder = NlEncoder(args)
        self.layernorm = LayerNorm(self.embedding_size)
        self.activate = GELU()
        self.copy = CopyNet(args)
        self.copy2 = CopyNet(args)
        self.dropout = nn.Dropout(p=0.1)
        self.depthembedding = nn.Embedding(40, self.embedding_size, padding_idx=0)
        self.gcnnm = GCNNM(self.embedding_size)
        self.position = PositionalEmbedding(self.embedding_size)

    def getBleu(self, losses, ngram):
        bleuloss = mindspore.nn.MaxPool1d(losses.unsqueeze(1), ngram, 1).squeeze(axis=1)
        bleuloss = mindspore.ops.sum(bleuloss, dim=-1)
        return bleuloss

    def construct(self, inputnl, inputnlchar, inputrule, inputruleparent, inputrulechild, inputParent, inputParentPath,
                  inputdepth, tmpf, tmpc, tmpindex, rulead, antimask, inputRes=None, mode="train"):
        selfmask = antimask
        # selfmask = antimask.unsqueeze(0).tile((inputtype.size(0), 1, 1)).unsqueeze(1)
        # admask = admask.unsqueeze(0).tile((inputtype.size(0), 1, 1)).float()
        rulemask = mindspore.ops.gt(inputrule, 0)
        inputParent = inputParent.float()
        # encode_nl
        nlencode, nlmask = self.encoder(inputnl, inputnlchar)
        # encode_rule
        childEm = self.rule_token_embedding(tmpc)
        childEm = self.conv(childEm.permute(0, 3, 1, 2))
        childEm = childEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        childEm = self.layernorm(childEm)
        fatherEm = self.rule_token_embedding(tmpf)
        ruleEmCom = self.rule_conv(mindspore.ops.stack([fatherEm, childEm], axis=-2).permute(0, 3, 1, 2))
        ruleEmCom = self.layernorm(ruleEmCom.permute(0, 2, 3, 1).squeeze(axis=-2))
        x = self.rule_embedding(tmpindex[0])
        #        for i in range(9):
        #            x = self.gcnnm(x, rulead[0], ruleEmCom[0]).view(self.resLen, self.embedding_size)
        ruleEm = self.rule_embedding(inputrule)
        ruleselect = x
        # print(inputdepth.shape)
        # depthEm = self.depthembedding(inputdepth.long())
        # depthEm = self.depth_conv(depthEm.permute(0, 3, 1, 2))
        # depthEm = depthEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        # depthEm = self.layernorm(depthEm)
        Ppath = self.rule_token_embedding(inputrulechild)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(axis=-2)
        ppathEm = self.layernorm(ppathEm)
        x = self.dropout(ruleEm + self.position(inputrule))
        for trans in self.encodeTransformerBlock:
            x = trans(x, selfmask, nlencode, nlmask, ppathEm, inputParent)
        decode = x
        # ppath
        Ppath = self.rule_token_embedding(inputParentPath)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(axis=-2)
        ppathEm = self.layernorm(ppathEm)
        x = self.dropout(ppathEm + self.position(inputrule))
        for trans in self.decodeTransformerBlocksP:
            x = trans(x, rulemask, decode, antimask, nlencode, nlmask)
        decode = x
        # genP1, _ = self.copy2(ruleselect.unsqueeze(0), decode)
        # resSoftmax = F.softmax(genP, dim=-1)
        genP, prob = self.copy(nlencode, decode)
        copymask = nlmask.unsqueeze(1).tile((1, inputrule.size(1), 1))
        genP = genP.masked_fill(copymask == 0, -1e9)
        # genP = torch.cat([genP1, genP], dim=2)
        # genP = F.softmax(genP, dim=-1)
        x = self.finalLinear(decode)
        x = self.activate(x)
        x = self.resLinear(x)
        resSoftmax = mindspore.ops.softmax(x, axis=-1)
        resSoftmax = resSoftmax * prob[:, :, 0].unsqueeze(axis=-1)
        genP = genP * prob[:, :, 1].unsqueeze(axis=-1)
        resSoftmax = mindspore.ops.cat([resSoftmax, genP], -1)
        if mode != "train":
            return resSoftmax
        resmask = mindspore.ops.gt(inputRes, 0)
        loss = -mindspore.ops.log(
            mindspore.ops.gather_elements(resSoftmax, -1, inputRes.unsqueeze(axis=-1)).squeeze(axis=-1))
        loss = loss.masked_fill(resmask == 0, 0.0)
        resTruelen = mindspore.ops.sum(resmask, dim=-1).float()
        totalloss = mindspore.ops.mean(loss, axis=-1) * self.code_len / resTruelen
        totalloss = totalloss  # + (self.getBleu(loss, 2) + self.getBleu(loss, 3) + self.getBleu(loss, 4)) / resTruelen
        # totalloss = torch.mean(totalloss)
        return totalloss, resSoftmax


class JointEmbber(nn.Cell):

    def __init__(self, args):
        super(JointEmbber, self).__init__()
        self.embedding_size = args.embedding_size
        self.codeEncoder = AttEncoder(args)
        self.margin = args.margin
        self.nlEncoder = NlEncoder(args)
        self.poolConvnl = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.poolConvcode = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.maxPoolnl = nn.MaxPool1d(args.NlLen)
        self.maxPoolcode = nn.MaxPool1d(args.CodeLen)

    def scoring(self, qt_repr, cand_repr):
        sim = mindspore.ops.cosine_similarity(qt_repr, cand_repr)
        return sim

    def nlencoding(self, inputnl, inputnlchar):
        nl = self.nlEncoder(inputnl, inputnlchar)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        return nl

    def codeencoding(self, inputcode, inputcodechar, ad):
        code = self.codeEncoder(inputcode, inputcodechar, ad)
        code = self.maxPoolcode(self.poolConvcode(code.permute(0, 2, 1))).squeeze(-1)
        return code

    def construct(self, inputnl, inputnlchar, inputcode, inputcodechar, ad, inputcodeneg, inputcodenegchar, adneg):
        nl = self.nlEncoder(inputnl, inputnlchar)
        code = self.codeEncoder(inputcode, inputcodechar, ad)
        codeneg = self.codeEncoder(inputcodeneg, inputcodenegchar, adneg)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        code = self.maxPoolcode(self.poolConvcode(code.permute(0, 2, 1))).squeeze(-1)
        codeneg = self.maxPoolcode(self.poolConvcode(codeneg.permute(0, 2, 1))).squeeze(-1)
        good_score = self.scoring(nl, code)
        bad_score = self.scoring(nl, codeneg)
        loss = (self.margin - good_score + bad_score).clamp(min=1e-6).mean()
        return loss, good_score, bad_score
