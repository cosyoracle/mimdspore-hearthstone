
import mindspore.dataset as ds
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import pickle
from ScheduledOptim import *
import torch.nn.functional as F


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'NlLen': 50,
    'CodeLen': 200,
    'batch_size': 32,
    'embedding_size': 312,
    'WoLen': 15,
    'Vocsize': 100,
    'Nl_Vocsize': 100,
    'max_step': 3,
    'margin': 0.5,
    'poolsize': 50,
    'Code_Vocsize': 100,
    'num_steps': 50,
    'rulenum': 10,
    'seed': 0
})


def save_model(model, dirs='checkpointSearch/'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    mindspore.save_checkpoint(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs='checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    # cprint(dirs)
    model.load_state_dict(mindspore.load(dirs + 'best_model.ckpt'))


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = mindspore.Tensor.from_numpy(data)
    else:
        assert isinstance(tensor, mindspore.Tensor)
    return tensor


def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    return ans


def getAdMask(size):
    ans = np.zeros([size, size])
    for i in range(size - 1):
        ans[i, i + 1] = 1.0
    return ans


def getRulePkl(vds):
    inputruleparent = []
    inputrulechild = []
    for i in range(len(vds.ruledict)):
        rule = vds.rrdict[i].strip().lower().split()
        inputrulechild.append(vds.pad_seq(vds.Get_Em(rule[2:], vds.Code_Voc), vds.Char_Len))
        inputruleparent.append(vds.Code_Voc[rule[0].lower()])
    return np.array(inputruleparent), np.array(inputrulechild)


def evalacc(model, dev_set):
    antimask = gVar(getAntiMask(args.CodeLen))
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).tile((2, 1)).long()
    tmpc = gVar(b).unsqueeze(0).tile((2, 1, 1)).long()
    devloader = mindspore.dataset.GeneratorDataset(source=dev_set,
                                                   shuffle=False,
                                                   column_names=['train_path', 'val_path', 'test_path', 'Nl_Voc',
                                                                 'Code_Voc', 'Char_Voc', 'Nl_Len', 'Code_Len',
                                                                 'Char_Len']).batch(batch_size=22)
    model = model.set_train(False)
    accs = []
    tcard = []
    antimask2 = antimask.unsqueeze(0).tile((22, 1, 1)).unsqueeze(1)
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).tile((2, 1, 1))
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).tile((2, 1)).long()
    for devBatch in tqdm(devloader):
        for i in range(len(devBatch)):
            devBatch[i] = gVar(devBatch[i])
        _, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[6], devBatch[7],
                       devBatch[8], tmpf, tmpc, tmpindex, rulead, antimask2, devBatch[5])
        pred = pre.argmax(dim=-1)
        resmask = mindspore.ops.gt(devBatch[5], 0)
        acc = (mindspore.ops.equal(pred, devBatch[5]) * resmask).float()  # .mean(dim=-1)
        predres = (1 - acc) * pred.float() * resmask.float()
        accsum = mindspore.ops.sum(acc, dim=-1)
        resTruelen = mindspore.ops.sum(resmask, dim=-1).float()
        print(mindspore.ops.equal(accsum, resTruelen))
        cnum = (mindspore.ops.equal(accsum, resTruelen)).sum().float()
        acc = acc.sum(dim=-1) / resTruelen
        accs.append(acc.mean().item())
        tcard.append(cnum.item())
        # print(devBatch[5])
        # print(predres)

    tnum = np.sum(tcard)
    acc = np.mean(accs)
    # wandb.log({"accuracy":acc})
    return acc, tnum


def train():
    mindspore.set_seed(args.seed)
    np.random.seed(args.seed)
    train_set = SumDataset(args, "train")
    print(len(train_set.rrdict))
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).tile((2, 1)).long()
    tmpc = gVar(b).unsqueeze(0).tile((2, 1, 1)).long()
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).tile((2, 1, 1))
    tmpindex = gVar(np.arange(len(train_set.ruledict))).unsqueeze(0).tile((2, 1)).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    dev_set = SumDataset(args, "val")
    test_set = SumDataset(args, "test")
    data_loader = ds.GeneratorDataset(source=train_set,
                                      shuffle=True, column_names=['train_path', 'val_path', 'test_path', 'Nl_Voc',
                                                                  'Code_Voc', 'Char_Voc', 'Nl_Len', 'Code_Len',
                                                                  'Char_Len']).batch(batch_size=args.batch_size)
    model = Decoder(args)
    # load_model(model)
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=1e-4)
    optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxAcc = 0
    maxC = 0
    maxAcc2 = 0
    maxC2 = 0

    model = model.set_train()
    antimask = gVar(getAntiMask(args.CodeLen))
    # model.to()

    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):
            if j % 3000 == 0:
                acc, tnum = evalacc(model, dev_set)
                acc2, tnum2 = evalacc(model, test_set)
                print("for dev " + str(acc) + " " + str(tnum) + " max is " + str(maxC))
                print("for test " + str(acc2) + " " + str(tnum2) + " max is " + str(maxC2))
                if maxC < tnum or maxC == tnum and maxAcc < acc:
                    maxC = tnum
                    maxAcc = acc
                    print("find better acc " + str(maxAcc))
                    save_model(model.module, 'checkpointSearch/')
                if maxC2 < tnum2 or maxC2 == tnum2 and maxAcc2 < acc2:
                    maxC2 = tnum2
                    maxAcc2 = acc2
                    print("find better acc " + str(maxAcc2))
                    save_model(model.module, "test%s/" % args.seed)
            # exit(0)
            antimask2 = antimask.unsqueeze(0).tile((args.batch_size, 1, 1)).unsqueeze(1)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[6], dBatch[7], dBatch[8],
                            tmpf, tmpc, tmpindex, rulead, antimask2, dBatch[5])
            loss = mindspore.ops.mean(loss) + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(
                0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(
                0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            j += 1


class Node:
    def __init__(self, name, d):
        self.name = name
        self.depth = d
        self.father = None
        self.child = []
        self.sibiling = None
        self.expanded = False
        self.fatherlistID = 0





beamss = []

if __name__ == "__main__":
    train()
