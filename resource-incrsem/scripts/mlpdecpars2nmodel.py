import sys, configparser, torch, re, os, time
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
import pdb


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def prepare_data():
    data = [line.strip() for line in sys.stdin]

    catBases, catAntes, hvBases, hvAntes, hvBaseFirsts, hvAnteFirsts, wordDists, sqWordDists, corefOns, labels = ([] for _ in range(10))
    #depth, catBase, hvBase, hvFiller, fDecs, hvBFirst, hvFFirst = ([] for _ in range(8))
    for line in data:
        catBase, catAnte, hvBase, hvAnte, wordDist, sqWordDist, corefOn, _, label = line.split(" ")
        #d, cb, hvb, hvf, fd = line.split(" ")
        #depth.append(int(d))
        #catBase.append(cb)
        #hvBase.append(hvb)
        #hvFiller.append(hvf)
        #fDecs.append(fd)
        catBases.append(catBase)
        catAntes.append(catAnte)
        hvBases.append(hvBase)
        hvAntes.append(hvAnte)
        wordDists.append(int(wordDist))
        sqWordDists.append(int(sqWordDist))
        corefOns.append(int(corefOn))
        labels.append(int(label))

    eprint("Linesplit complete")
    # Extract first KVec from sparse HVec
    for hvec in hvBases:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvBaseFirsts.append(match[0].split(","))
    eprint("hvBaseFirsts ready")

    for hvec in hvAntes:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvAnteFirsts.append(match[0].split(","))
    eprint("hvAnteFirsts ready")

    # Mapping from category & HVec to index
    flat_hvB = [hvec for sublist in hvBaseFirsts for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    flat_hvA = [hvec for sublist in hvAnteFirsts for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    allCats = set(catBases).union(set(catAntes))
    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(allCats)))}
    #fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(set(fDecs)))}
    hvec_to_ix = {hvec: i for i, hvec in enumerate(sorted(set(flat_hvB + flat_hvA)))}

    cat_base_ixs = [cat_to_ix[cat] for cat in catBases]
    cat_ante_ixs = [cat_to_ix[cat] for cat in catAntes]
    #fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    hvb_row, hvb_col, hvb_top, hva_row, hva_col, hva_top = ([] for _ in range(6))

    # KVec index sparse matrix and "Top" KVec counts
    for i, sublist in enumerate(hvBaseFirsts):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvb_row.append(i)
                hvb_col.append(hvec_to_ix[hvec])
        hvb_top.append([top_count])
    hvb_mat = csr_matrix((np.ones(len(hvb_row), dtype=np.int32), (hvb_row, hvb_col)),
                         shape=(len(hvBaseFirsts), len(hvec_to_ix)))
    eprint("hvb_mat ready")

    for i, sublist in enumerate(hvAnteFirsts):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hva_row.append(i)
                hva_col.append(hvec_to_ix[hvec])
        hva_top.append([top_count])
    hva_mat = csr_matrix((np.ones(len(hva_row), dtype=np.int32), (hva_row, hva_col)),
                         shape=(len(hvAnteFirsts), len(hvec_to_ix)))
    eprint("hva_mat ready")

    eprint("Number of input KVecs: {}".format(len(hvec_to_ix)))
    #eprint("Number of output F categories: {}".format(len(fdecs_to_ix)))

    return cat_to_ix, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvec_to_ix, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels 
    #return depth, cat_b_ix, hvb_mat, hvf_mat, cat_to_ix, fdecs_ix, fdecs_to_ix, hvec_to_ix, hvb_top, hvf_top


def prepare_data_dev(dev_decpars_file, cat_to_ix, fdecs_to_ix, hvec_to_ix):
    #TODO not supported yet
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, catBase, hvBase, hvFiller, fDecs, hvBFirst, hvFFirst = ([] for _ in range(7))

    for line in data:
        d, cb, hvb, hvf, fd = line.split(" ")
        if cb not in cat_to_ix or fd not in fdecs_to_ix:
            continue
        depth.append(int(d))
        catBase.append(cb)
        hvBase.append(hvb)
        hvFiller.append(hvf)
        fDecs.append(fd)

    for kvec in hvBase:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvBFirst.append(match[0].split(","))

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append(match[0].split(","))

    cat_b_ix = [cat_to_ix[cat] for cat in catBase]
    fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    hvb_row, hvb_col, hvf_row, hvf_col, hvb_top, hvf_top = ([] for _ in range(6))

    # KVec indices and "Top" KVec counts
    for i, sublist in enumerate(hvBFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvb_row.append(i)
                hvb_col.append(hvec_to_ix[hvec])
        hvb_top.append([top_count])
    hvb_mat = csr_matrix((np.ones(len(hvb_row), dtype=np.int32), (hvb_row, hvb_col)),
                         shape=(len(hvBFirst), len(hvec_to_ix)))

    for i, sublist in enumerate(hvFFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvf_row.append(i)
                hvf_col.append(hvec_to_ix[hvec])
        hvf_top.append([top_count])
    hvf_mat = csr_matrix((np.ones(len(hvf_row), dtype=np.int32), (hvf_row, hvf_col)),
                         shape=(len(hvFFirst), len(hvec_to_ix)))

    return depth, cat_b_ix, hvb_mat, hvf_mat, fdecs_ix, hvb_top, hvf_top


class NModel(nn.Module):

    def __init__(self, cat_vocab_size, hvec_vocab_size, syn_size, sem_size, hidden_dim, output_dim):
        super(NModel, self).__init__()
        self.hvec_vocab_size = hvec_vocab_size
        self.sem_size = sem_size
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hvec_embeds = nn.Embedding(hvec_vocab_size, sem_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(2*syn_size+2*sem_size+3, self.hidden_dim, bias=True)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    #def forward(self, cat_base_ixs, cat_ante_ixs, hvbases, hvantes, worddists, sqworddists, corefons, use_gpu):
    def forward(self, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvb_top, hva_top, worddists, sqworddists, corefons, use_gpu, ablate_sem):
    #def forward(self, d_onehot, cat_b_ix, hvb_mat, hvf_mat, hvb_top, hvf_top, use_gpu, ablate_sem):
        cat_base_embed = self.cat_embeds(cat_base_ixs)
        cat_ante_embed = self.cat_embeds(cat_ante_ixs)
        hvb_top = torch.FloatTensor(hvb_top)
        hva_top = torch.FloatTensor(hva_top)

        if use_gpu >= 0:
            cat_base_embed = cat_base_embed.to("cuda")
            cat_ante_embed = cat_ante_embed.to("cuda")
            hvb_mat = hvb_mat.to("cuda")
            hva_mat = hva_mat.to("cuda")
            hvb_top = hvb_top.to("cuda")
            hva_top = hva_top.to("cuda")

        if ablate_sem:
            hvb_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float) + hvb_top
            hva_embed = torch.zeros([hva_top.shape[0], self.sem_size], dtype=torch.float) + hvf_top

        else:
            hvb_mat = hvb_mat.tocoo()
            hvb_mat = torch.sparse.FloatTensor(torch.LongTensor([hvb_mat.row.tolist(), hvb_mat.col.tolist()]),
                                               torch.FloatTensor(hvb_mat.data.astype(np.float32)),
                                               torch.Size(hvb_mat.shape))
            hva_mat = hva_mat.tocoo()
            hva_mat = torch.sparse.FloatTensor(torch.LongTensor([hva_mat.row.tolist(), hva_mat.col.tolist()]),
                                               torch.FloatTensor(hva_mat.data.astype(np.float32)),
                                               torch.Size(hva_mat.shape))
            hvb_embed = torch.sparse.mm(hvb_mat, self.hvec_embeds.weight) + hvb_top
            hva_embed = torch.sparse.mm(hva_mat, self.hvec_embeds.weight) + hva_top
        x = torch.cat((cat_base_embed, cat_ante_embed, hvb_embed, hva_embed, worddists.unsqueeze(dim=1), sqworddists.unsqueeze(dim=1), corefons.unsqueeze(dim=1)), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, hidden_dim, 
          num_epochs, batch_size, learning_rate, weight_decay, l2_reg, 
          ablate_sem, useClassFreqWeighting):
    #depth, cat_b_ix, hvb_mat, hvf_mat, cat_to_ix, fdecs_ix, fdecs_to_ix, hvec_to_ix, hvb_top, hvf_top = prepare_data()
    cat_to_ix, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvec_to_ix, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels = prepare_data() 
    #depth = F.one_hot(torch.LongTensor(depth), 7).float()
    #cat_b_ix = torch.LongTensor(cat_b_ix)
    #target = torch.LongTensor(fdecs_ix)
    #model = FModel(len(cat_to_ix), len(hvec_to_ix), syn_size, sem_size, hidden_dim, len(fdecs_to_ix))
    cat_base_ixs = torch.LongTensor(cat_base_ixs)
    cat_ante_ixs = torch.LongTensor(cat_ante_ixs)
    #hvBases = torch.FloatTensor(hvBases)
    #hvAntes = torch.FloatTensor(hvAntes)
    wordDists = torch.LongTensor(wordDists)
    sqWordDists = torch.LongTensor(sqWordDists)
    corefOns = torch.LongTensor(corefOns)
    target = torch.LongTensor(labels)
    outputdim = len(set(target.tolist())) 
    assert outputdim == 2
    model = NModel(len(cat_to_ix), len(hvec_to_ix), syn_size, sem_size, hidden_dim, outputdim) #output_dim is last param

    if use_gpu >= 0:
        #depth = depth.to("cuda")
        cat_base_ixs = cat_base_ixs.to("cuda")
        cat_ante_ixs = cat_ante_ixs.to("cuda")
        #hvBases = hvBases.to("cuda")
        #hvAntes = hvAntes.to("cuda")
        wordDists = wordDists.to("cuda")
        sqWordDists = sqWordDists.to("cuda")
        corefOns = corefOns.to("cuda")
        target = target.to("cuda")
        model = model.cuda()
        #cat_base_ixs = cat_base_ixs.to("cuda")
        #target = target.to("cuda")
        #model = model.cuda()

    if use_dev >= 0:
        #TODO not implemented yet
        dev_depth, dev_cat_b_mat, dev_hvb_mat, dev_hvf_mat, dev_fdecs_ix, dev_hvb_top, dev_hvf_top = prepare_data_dev(
            dev_decpars_file, cat_to_ix, fdecs_to_ix, hvec_to_ix)
        dev_depth = F.one_hot(torch.LongTensor(dev_depth), 7).float()
        dev_cat_b_ix = torch.LongTensor(dev_cat_b_ix)
        dev_target = torch.LongTensor(dev_fdecs_ix)

        if use_gpu >= 0:
            dev_depth = dev_depth.to("cuda")
            dev_cat_b_ix = dev_cat_b_ix.to("cuda")
            dev_target = dev_target.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #TODO implement useClassFreqWeighting
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start NModel training...")
    epoch = 0

    while True:
        c0 = time.time()
        model.train()
        epoch += 1
        permutation = torch.randperm(len(target))
        total_train_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, len(target), batch_size):
            indices = permutation[i:i + batch_size]

            batch_catbase, batch_catante, batch_worddist, batch_sqworddist, batch_corefon, batch_target = cat_base_ixs[indices], cat_ante_ixs[indices], wordDists[indices], sqWordDists[indices], corefOns[indices], target[indices]
            #batch_d, batch_c, batch_target = depth[indices], cat_b_ix[indices], target[indices]
            batch_hvb_mat, batch_hva_mat = hvb_mat[np.array(indices), :], hva_mat[np.array(indices), :]
            batch_hvb_top, batch_hva_top = [hvb_top[i] for i in indices], [hva_top[i] for i in indices]
            if use_gpu >= 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])
            for param in model.parameters():
                l2_loss += torch.mean(param.pow(2))

            #output = model(batch_d, batch_c, batch_hvb_mat, batch_hvf_mat, batch_hvb_top, batch_hvf_top, use_gpu,
            output = model(batch_catbase, batch_catante, batch_hvb_mat, 
                           batch_hva_mat, batch_hvb_top, batch_hva_top, 
                           batch_worddist.float(), batch_sqworddist.float(), 
                           batch_corefon.float(), use_gpu, ablate_sem)
            _, ndec = torch.max(output.data, 1)
            train_correct = (ndec == batch_target).sum().item()
            total_train_correct += train_correct
            nll_loss = criterion(output, batch_target)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            #TODO not implemented yet
            with torch.no_grad():
                dev_pred = model(dev_depth, dev_cat_b_ix, dev_hvb_mat, dev_hvf_mat, dev_hvb_top, dev_hvf_top, use_gpu,
                                 ablate_sem)
                _, dev_fdec = torch.max(dev_pred.data, 1)
                dev_correct = (dev_fdec == dev_target).sum().item()
                dev_loss = criterion(dev_pred, dev_target)
                total_dev_loss += dev_loss.item()
                dev_acc = 100 * (dev_correct / len(dev_depth))
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".
               format(epoch, total_train_loss / (len(target) // batch_size), 100 * (total_train_correct / len(target)),
                      total_dev_loss, dev_acc, time.time() - c0))

        if epoch == num_epochs:
            break

    #return model, cat_to_ix, fdecs_to_ix, hvec_to_ix
    return model, cat_to_ix, hvec_to_ix


def main(config):
    n_config = config["NModel"]
    model, cat_to_ix, hvec_to_ix = train(n_config.getint("Dev"), 
                                   n_config.get("DevFile"), 
                                   n_config.getint("GPU"), 
                                   n_config.getint("SynSize"), 
                                   n_config.getint("SemSize"), 
                                   n_config.getint("HiddenSize"), 
                                   n_config.getint("NEpochs"), 
                                   n_config.getint("BatchSize"), 
                                   n_config.getfloat("LearningRate"), 
                                   n_config.getfloat("WeightDecay"), 
                                   n_config.getfloat("L2Reg"), 
                                   n_config.getboolean("AblateSem"),
                                   n_config.getboolean("UseClassFreqWeighting"))

    if n_config.getint("GPU") >= 0:
        cat_embeds = list(model.parameters())[0].data.cpu().numpy()
        hvec_embeds = list(model.parameters())[1].data.cpu().numpy()
        first_weights = list(model.parameters())[2].data.cpu().numpy()
        first_biases = list(model.parameters())[3].data.cpu().numpy()
        second_weights = list(model.parameters())[4].data.cpu().numpy()
        second_biases = list(model.parameters())[5].data.cpu().numpy()
    else:
        cat_embeds = list(model.parameters())[0].data.numpy()
        hvec_embeds = list(model.parameters())[1].data.numpy()
        first_weights = list(model.parameters())[2].data.numpy()
        first_biases = list(model.parameters())[3].data.numpy()
        second_weights = list(model.parameters())[4].data.numpy()
        second_biases = list(model.parameters())[5].data.numpy()

    eprint(first_weights.shape, second_weights.shape)
    print("N F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("N f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("N S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("N s " + ",".join(map(str, second_biases.flatten('F').tolist())))
    for cat, ix in sorted(cat_to_ix.items()):
        print("C " + str(cat) + " [" + ",".join(map(str, cat_embeds[ix])) + "]")
    if not n_config.getboolean("AblateSem"):
        for hvec, ix in sorted(hvec_to_ix.items()):
            print("K " + str(hvec) + " [" + ",".join(map(str, hvec_embeds[ix])) + "]")
    #for fdec, ix in sorted(fdecs_to_ix.items()):
    #    print("f " + str(ix) + " " + str(fdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
