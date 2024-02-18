'''
@File       :   train_hinge.py
@Time       :   2023/02/04 10:51:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Train reward model.
'''
import sys

sys.path.append(f'/home/quickjkee/diversity/models/src/config')
sys.path.append(f'/home/quickjkee/diversity/models/src')
sys.path.append(f'/home/quickjkee/diversity/models')

# LOCAL
from models.src.config.options import *
from models.src.config.utils import *
from models.src.config.learning_rates import get_learning_rate_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num

# GLOBAL
import torch
import math
import numpy as np
import torch.nn as nn
import sys
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, Sampler
from torch.backends import cudnn
from metrics import samples_metric
from catalyst.data.sampler import DistributedSamplerWrapper
from scipy import stats

def std_log():
    if get_rank() == 0:
        save_path = make_path()
        makedir(config['log_base'])
        sys.stdout = open(os.path.join(config['log_base'], "{}.txt".format(save_path)), "w")


def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def pearson(x, y):
    x = np.array(x)
    y = np.array(y)
    res = stats.spearmanr(x, y)
    return res.statistic

# def loss_func(predict, target):
#    loss = nn.CrossEntropyLoss(reduction='none')
#    loss_list = loss(predict, target)
#    loss = torch.mean(loss_list, dim=0)
#    correct = torch.eq(torch.max(F.softmax(predict, dim=1), dim=1)[1], target).view(-1)
#    acc = torch.sum(correct).item() / len(target)
#    return loss, loss_list, acc


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        print(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss, ce_loss


def samples_metric_thresh(labels_sim, labels_true):
    accs = []
    threshs = np.linspace(min(labels_sim), max(labels_sim), 10)
    for thresh in threshs:
        curr_pred = (np.array(labels_sim) > thresh) * 1
        curr_pred = list(curr_pred.astype(int))
        accs.append(samples_metric(labels_true, curr_pred)[0])
    return max(accs)


def run_train(train_dataset,
              valid_dataset,
              model,
              label,
              loss_w):
    if opts.std_log:
        std_log()

    if opts.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(opts.seed + local_rank)

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(opts.seed)

    writer = visualizer()
    model.to(device)
    model.device = device
    loss_fn = FocalLoss(alpha=loss_w.to(device) * 500)
    test_dataset = valid_dataset

    def loss_cl_fn(predict, target):
        loss, loss_list = loss_fn(predict, target)
        preds_probs = F.softmax(predict, dim=1)
        correct = torch.eq(torch.max(preds_probs, dim=1)[1], target).view(-1)
        acc = torch.sum(correct).item() / len(target)

        return loss, acc

    def loss_sim_fn(img_1, img_2, target):
        sim = F.cosine_similarity(img_1, img_2, dim=1) #torch.abs(img_1 - img_2).sum(dim=1)
        loss = 1 - sim
        new_target = target * (2) - 1
        l = nn.HingeEmbeddingLoss(margin=0.05)
        sim_loss = l(loss, new_target)
#        print(sim)
#        print(target)
#        print('====')
        with torch.no_grad():
            sim_list = (sim + 1) / 2
        return sim_loss, sim_list

    if opts.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn if not opts.rank_pair else None)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                  collate_fn=collate_fn if not opts.rank_pair else None)

    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True,
                              collate_fn=collate_fn if not opts.rank_pair else None)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True,
                             collate_fn=collate_fn if not opts.rank_pair else None)

    # Set the training iterations.
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("len(train_loader) = ", len(train_loader))
    print("len(test_dataset) = ", len(test_dataset))
    print("len(test_loader) = ", len(test_loader))
    print("steps_per_valid = ", steps_per_valid)

    if opts.preload_path:
        model = preload_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2),
                                 eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # valid result print and log
    if get_rank() == 0:
        model.eval()
        valid_loss, valid_loss_sim = 0, 0
        labels_preds, labels_true, labels_sim = [], [], []
        with torch.no_grad():
            for step, batch_data_package in enumerate(valid_loader):
                predict, emb_img_1, emb_img_2 = model(batch_data_package)
                target = batch_data_package[label].to(device)
                loss_cl, acc = loss_cl_fn(predict, target)
                loss_sim, sim_list = loss_sim_fn(emb_img_1, emb_img_2, target)

                # RECORDING
                # --------------------------
                # Labels for further prediction
                labels_preds.append(torch.max(F.softmax(predict, dim=1), dim=1)[1].cpu().view(-1).int())
                labels_true.append(target.cpu().view(-1).int())
                labels_sim.append(F.softmax(predict, dim=1)[:,1].cpu().view(-1))

                # losses
                valid_loss += loss_cl.cpu().item()
                valid_loss_sim += loss_sim.cpu().item()
                # --------------------------

        # AGGREGATION
        # --------------------------
        labels_preds = list(np.array(torch.cat(labels_preds, 0)))
        labels_true = list(np.array(torch.cat(labels_true, 0)))
        labels_sim = list(np.array(torch.cat(labels_sim, 0)))

        valid_loss = valid_loss / len(valid_loader)
        valid_loss_sim = valid_loss_sim / len(valid_loader)
        boots_acc = samples_metric(labels_true, labels_preds)[0]
        sim_boots_acc = pearson(labels_true, labels_sim)
        # --------------------------

        print('Validation - Iteration %d | Loss %6.5f | SimLoss %6.5f | BootsAcc %6.4f | Pearson %6.4f' %
              (0, valid_loss, valid_loss_sim, boots_acc, sim_boots_acc))
        writer.add_scalar('Validation-Loss', valid_loss, global_step=0)
        writer.add_scalar('Validation-BootsAcc', boots_acc, global_step=0)

    best_acc = 0
    optimizer.zero_grad()
    losses = []
    acc_list = []
    for epoch in range(opts.epochs):

        for step, batch_data_package in enumerate(train_loader):
            model.train()

            # Predict
            predict, emb_img_1, emb_img_2 = model(batch_data_package)
            target = batch_data_package[label].to(device)
            loss_cl, acc = loss_cl_fn(predict, target)
            loss_sim, sim_list = loss_sim_fn(emb_img_1, emb_img_2, target)
            loss = loss_cl #+ 10 * loss_sim
            # loss regularization
            loss = loss / opts.accumulation_steps
            # back propagation
            loss.backward()

            losses.append(loss.detach().cpu().item())
            acc_list.append(acc)

            iterations = epoch * len(train_loader) + step + 1
            train_iteration = iterations / opts.accumulation_steps

            # update parameters of net
            if (iterations % opts.accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # train result print and log 
                if get_rank() == 0:
                    losses_lg = np.mean(losses)
                    print('Iteration %d | Loss %6.5f | Acc %6.4f' %
                          (train_iteration, losses_lg, sum(acc_list) / len(acc_list)))
                    writer.add_scalar('Train-Loss', losses_lg, global_step=train_iteration)
                    writer.add_scalar('Train-Acc', sum(acc_list) / len(acc_list), global_step=train_iteration)

                losses.clear()
                acc_list.clear()

            # VALIDATION
            # ----------------------------------------------------------------------------
            if (iterations % steps_per_valid) == 0:
                if get_rank() == 0:
                    model.eval()
                    valid_loss, valid_loss_sim = 0, 0
                    labels_preds, labels_true, labels_sim = [], [], []
                    with torch.no_grad():
                        for step, batch_data_package in enumerate(valid_loader):
                            predict, emb_img_1, emb_img_2 = model(batch_data_package)
                            target = batch_data_package[label].to(device)
                            loss_cl, acc = loss_cl_fn(predict, target)
                            loss_sim, sim_list = loss_sim_fn(emb_img_1, emb_img_2, target)

                            # RECORDING
                            # --------------------------
                            # Labels for further prediction
                            labels_preds.append(torch.max(F.softmax(predict, dim=1), dim=1)[1].cpu().view(-1).int())
                            labels_true.append(target.cpu().view(-1).int())
                            labels_sim.append(F.softmax(predict, dim=1)[:,1].cpu().view(-1))

                            # losses
                            valid_loss += loss_cl.cpu().item()
                            valid_loss_sim += loss_sim.cpu().item()
                            # --------------------------

                    # AGGREGATION
                    # --------------------------
                    labels_preds = list(np.array(torch.cat(labels_preds, 0)))
                    labels_true = list(np.array(torch.cat(labels_true, 0)))
                    labels_sim = list(np.array(torch.cat(labels_sim, 0)))

                    valid_loss = valid_loss / len(valid_loader)
                    valid_loss_sim = valid_loss_sim / len(valid_loader)
                    boots_acc = samples_metric(labels_true, labels_preds)[0]
                    sim_boots_acc = pearson(labels_sim, labels_true)
                    # --------------------------

                    print(
                        'Validation - Iteration %d | Loss %6.5f | SimLoss %6.5f | BootsAcc %6.4f | Pearson %6.4f'
                        % (0, valid_loss, valid_loss_sim, boots_acc, sim_boots_acc))
                    writer.add_scalar('Validation-Loss', valid_loss, global_step=0)
                    writer.add_scalar('Validation-BootsAcc', boots_acc, global_step=0)

                    if boots_acc > best_acc:
                        print("Best BootsAcc so far. Saving model")
                        best_acc = boots_acc
                        print("best_acc = ", best_acc)
                        save_model(model)
            # ----------------------------------------------------------------------------

    # test model
    if get_rank() == 0:
        print("training done")
        print("test: ")
        model = load_model(model)
        model.eval()

        test_loss = 0
        labels_preds, labels_true, labels_sim = [], [], []
        with torch.no_grad():
            for step, batch_data_package in enumerate(test_loader):
                predict, emb_img_1, emb_img_2 = model(batch_data_package)
                target = batch_data_package[label].to(device)
                loss_cl, acc = loss_cl_fn(predict, target)
                loss_sim, sim_list = loss_sim_fn(emb_img_1, emb_img_2, target)

                # RECORDING
                # --------------------------
                # Labels for further prediction
                labels_preds.append(torch.max(F.softmax(predict, dim=1), dim=1)[1].cpu().view(-1).int())
                labels_true.append(target.cpu().view(-1).int())
                labels_sim.append(F.softmax(predict, dim=1)[:,1].cpu().view(-1))

                # losses
                test_loss += loss_cl.cpu().item()
                # --------------------------

        # AGGREGATION
        # --------------------------
        labels_preds = list(np.array(torch.cat(labels_preds, 0)))
        labels_true = list(np.array(torch.cat(labels_true, 0)))
        labels_sim = list(np.array(torch.cat(labels_sim, 0)))

        test_loss = test_loss / len(test_loader)
        boots_acc = samples_metric(labels_true, labels_preds)[0]
        sim_boots_acc = pearson(labels_sim, labels_true)
        # --------------------------

        print('Test Loss %6.5f | Acc %6.4f | Pearson %6.4f' % (test_loss, boots_acc, sim_boots_acc))
