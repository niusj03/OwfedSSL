import sys
import copy
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_cifar as datasets
import client_open_world_cifar as client_datasets
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, cluster_acc_w
from sklearn import metrics
import numpy as np
import os
# from utils_cluster import
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle


# Global settings #     #
is_track = True         #
# --------------- #     #

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def receive_models(clients_model):
    global uploaded_weights
    global uploaded_models
    uploaded_weights = []
    uploaded_models = []
    for model in clients_model:
        uploaded_weights.append( 1.0 / len(clients_model)) # each model is given equal importance in the aggregation process
        # self.uploaded_models.append(copy.deepcopy(client.model.parameters()))
        uploaded_models.append(model.parameters())

def add_parameters(w, client_model):
    for (name, server_param), client_param in zip(global_model.named_parameters(), client_model):
        if "centroids" not in name: # 'centroids', 'local centroids' and 'local labeled centroids'
            server_param.data += client_param.data.clone() * w
        if "local_labeled_centroids" in name:
            server_param.data += client_param.data.clone() * w

def aggregate_parameters():
    # resets the weights of global model to zero, except for the centroids and local_centroids
    for name, param in global_model.named_parameters():
        if "centroids" not in name:
            param.data = torch.zeros_like(param.data)
        if "local_labeled_centroids" in name:
            param.data = torch.zeros_like(param.data)
    for w, client_model in zip(uploaded_weights, uploaded_models):
        # update the global model with the weighted sum of the client models: backbone parameters and 
        add_parameters(w, client_model)

def args_setting():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--dataset', default='cifar10', help='dataset setting')
    parser.add_argument('--clients-num', default=5, type=int)
    parser.add_argument('--global-rounds', default=40, type=int)
    parser.add_argument('--labeled-num', default=5, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size')
    parser.add_argument('--gpu', default='0', type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda:"+args.gpu if args.cuda else "cpu")
    print("-------------------- use ",device,"-----------")
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    return args, device

def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer, client_id, global_round): # m refers to mean_uncert

    model.local_labeled_centroids.weight.data.zero_()  # model.local_labeled_centroids.weight.data: torch.Size([10, 512])
    labeled_samples_num = [0 for _ in range(10)]
    model.train()

    bce = nn.BCELoss()
    m = min(m, 0.5)
    # m = 0
    ce = MarginLoss(m=-1*m)
    unlabel_ce = MarginLoss(m=0) #(m=-1*m)
    unlabel_loader_iter = cycle(train_unlabel_loader)

    # ----- ################################### -----
    if is_track:
        bce_losses, ce_losses, entropy_losses  = AverageMeter('bce_loss', ':.4e'), AverageMeter('ce_loss', ':.4e'), AverageMeter('entropy_loss', ':.4e')
        conf_pred, conf_cls = AverageMeter("conf_pred", ":.4e"), AverageMeter("conf_cls", ":.4e")
        np_cluster_preds = np.array([]) # cluster_preds
        np_unlabel_targets = np.array([])

        # pred_prev, pred_sk = torch.tensor([]).cuda(), torch.tensor([]).cuda()
        # conf_prev, conf_swav = AverageMeter("conf_prev", ":.4e"), AverageMeter("conf_swav", ":.4e")
        # accept_id, accept_ood = 0, 0
        # all_label_in_Epoch = torch.tensor([]).cuda() # for baselinevv, this only recode unlabeled data
        # id_probs = AverageMeter("id_probs", ":.4e")
        # ood_probs = AverageMeter("ood_probs", ":.4e")
    # ----- ################################### -----

    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        
        ## 各个类的不确定性权重（固定值）
        beta = 0.2
        Nk = [1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600, 1600, 1600, 1600]
        Nmax = 1600 * 5
        p_weight = [beta**(1-Nk[i]/Nmax) for i in range(10)]
        

        ((ux, ux2), unlabel_target) = next(unlabel_loader_iter)
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        labeled_len = len(target)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()
        output, feat = model(x) # output: [batch size, 10]; feat: [batch size, 512]
        output2, feat2 = model(x2)
        prob = F.softmax(output, dim=1)
        reg_prob = F.softmax(output[labeled_len:], dim=1) # unlabel data's prob
        prob2 = F.softmax(output2, dim=1)
        reg_prob2 = F.softmax(output2[labeled_len:], dim=1)  # unlabel data's prob

        # Update local_labeled_centroids
        # step1: Count the number of each class (labeled data), and sum the feature vectors of each class
        for feature, true_label in zip(feat[:labeled_len].detach().clone(), target):
            labeled_samples_num[true_label] += 1
            model.local_labeled_centroids.weight.data[true_label] += feature
        # step2: Update feature vector for each class
        for idx, (feature_centroid, num) in enumerate(zip(model.local_labeled_centroids.weight.data, labeled_samples_num)): # idx from 0 to 9
            if num > 0:
                model.local_labeled_centroids.weight.data[idx] = feature_centroid/num

        C = model.centroids.weight.data.detach().clone().T # C: [512, 10]
        Z1 = F.normalize(feat, dim=1)
        Z2 = F.normalize(feat2, dim=1)
        cP1 = Z1 @ C # [Batch size, 10] cosine-similarity
        cP2 = Z2 @ C # [Batch size, 10]    
        tP1 = F.softmax(cP1 / model.T, dim=1)
        tP2 = F.softmax(cP2 / model.T, dim=1)



        # ------------------------------ Tracking ------------------------------
        # if is_track:
        #     conf_pred_b, target_cls_b = torch.max(prob[labeled_len:], dim = -1) # pred: [512]; target: [512]
        #     conf_cls_b, target_cls = torch.max(tP1[labeled_len:], dim = -1) # cluster_pred: [512]; target: [170]
        #     conf_pred.update(conf_pred_b.mean().item(), args.batch_size)
        #     conf_cls.update(conf_cls_b.mean().item(), args.batch_size)
        # -------------------- ############################# --------------------
            
        ################################# L_reg objective #################################
        # L_reg:  reg_prob中每一行的预测label
        if True:
            # abstract the predict labels for each unlabel data  
            copy_reg_prob1 = copy.deepcopy(reg_prob.detach())
            copy_reg_prob2 = copy.deepcopy(reg_prob2.detach())
            reg_label1 = np.argmax(copy_reg_prob1.cpu().numpy(), axis=1)
            reg_label2 = np.argmax(copy_reg_prob2.cpu().numpy(), axis=1)
            ### 制作target, target 除了 label=1 外与 reg_prob 一致
            # ajust copy_reg_prob: make the largest probability -> 1
            for idx, (label, oprob) in enumerate(zip(reg_label1, copy_reg_prob1)):
                copy_reg_prob1[idx][label] = 1
            for idx, (label, oprob) in enumerate(zip(reg_label2, copy_reg_prob2)):
                copy_reg_prob2[idx][label] = 1
                    
            L1_loss = nn.L1Loss()
            L_reg1, L_reg2 = 0.0, 0.0
            for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob, copy_reg_prob1, reg_label1)):
                L_reg1 += L1_loss(reg_prob[idx], copy_reg_prob1[idx]) * p_weight[label]
            for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob2, copy_reg_prob2, reg_label2)):
                L_reg2 += L1_loss(reg_prob2[idx], copy_reg_prob2[idx]) * p_weight[label]
            L_reg1 = L_reg1 / len(reg_label1)
            L_reg2 = L_reg2 / len(reg_label2)
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ L_reg objective ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
         
        ################################# ORCA objective #################################
        # ORCA objective include standard cross-entropy (labeled), BCE loss (labeled & unlabeled) and Entropy_loss (mean-distribution)
        if True:
            # calculate distance
            feat_detach = feat.detach()
            feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
            cosine_dist = torch.mm(feat_norm, feat_norm.t())

            pos_pairs = [] # closet index vectors: [512,1]
            target_np = target.cpu().numpy()
            # label part for pos_pairs
            for i in range(labeled_len):
                target_i = target_np[i]
                idxs = np.where(target_np == target_i)[0]
                if len(idxs) == 1:
                    pos_pairs.append(idxs[0])
                else:
                    selec_idx = np.random.choice(idxs, 1)
                    while selec_idx == i:
                        selec_idx = np.random.choice(idxs, 1)
                    pos_pairs.append(int(selec_idx))
            # unlabel part for pos_pairs
            unlabel_cosine_dist = cosine_dist[labeled_len:, :]
            vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
            pos_pairs.extend(pos_idx)

            # standard cross-entropy (labeled)
            ce_loss = ce(output[:labeled_len], target)

            # BCE loss (labeled & unlabeled)
            pos_prob = prob2[pos_pairs, :]
            pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
            ones = torch.ones_like(pos_sim)
            bce_loss = bce(pos_sim, ones)

            # Entropy_loss (mean-distribution)
            entropy_loss = entropy(torch.mean(prob, 0))
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ORCA objective ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

        ################################# CALIBRATION objective #################################
        # CALIBRATION objective includes 
        if True:
            confidence_cluster_pred, cluster_pred = tP1.max(1) # cluster_pred: [512]; target: [170]

            ### 统计 cluster_pred (伪标签，cluster id) 置信度 ###
            confidence_list = [0 for _ in range(10)]
            num_of_cluster = [0 for _ in range(10)]
            for confidence, cluster_id in zip(confidence_cluster_pred[labeled_len:], cluster_pred[labeled_len:]):
                confidence_list[cluster_id] += confidence
                num_of_cluster[cluster_id] += 1
            for cluster_id, (sum_confidence, num) in enumerate(zip(confidence_list, num_of_cluster)):
                if num > 0:
                    confidence_list[cluster_id] = np.around(confidence_list[cluster_id].cpu().detach().numpy()/ num, 4)
            
            threshold = 0.95
            confidence_mask = (confidence_cluster_pred[labeled_len:] > threshold)
            confidence_mask = torch.nonzero(confidence_mask)
            confidence_mask = torch.squeeze(confidence_mask)
            if client_id == 0:
                print("confidence_mask: ", confidence_mask.cpu().numpy())
            # sys.exit(0)
            print("global round: ", global_round, ";   client_id: ", client_id, ";   confidence_list: ", confidence_list)

            # Clustering loss
            # L_cluster = - torch.sum(tP1 * torch.log(tP2), dim=1).mean();  Clustering loss used in Orchestra (Euclidean distance is also utilized)
            cluster_pos_prob = tP2[pos_pairs, :] # cluster_pos_prob size: [512,10]
            L_cluster = - torch.sum(tP1 * torch.log(cluster_pos_prob), dim=1).mean() #[170(label)/512-170(unlabel)]
            
            # bce
            # cluster_pos_sim = torch.bmm(tP1.view(args.batch_size, 1, -1), cluster_pos_prob.view(args.batch_size, -1, 1)).squeeze()
            # cluster_ones = torch.ones_like(cluster_pos_sim)
            # cluster_bce_loss = bce(cluster_pos_sim, cluster_ones)

            # unlabel ce loss
            # unlabel_ce_loss = unlabel_ce(output[labeled_len:], cluster_pred[labeled_len:])
            unlabel_ce_loss = unlabel_ce(output[labeled_len:].index_select(0, confidence_mask) , cluster_pred[labeled_len:].index_select(0, confidence_mask))

            np_cluster_preds = np.append(np_cluster_preds, cluster_pred[labeled_len:].cpu().numpy())
            np_unlabel_targets = np.append(np_unlabel_targets, unlabel_target.cpu().numpy())
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ CALIBRATION objective ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        
        ################################# Final objective #################################
        if global_round > 4: #4
            if global_round > 6: #6
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster + unlabel_ce_loss # + 2 * L_reg1 + 2 * L_reg2  # + L_cluster # 调整L_reg倍率
            else:
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster # + 2 * L_reg1 + 2 * L_reg2 #+ L_cluster # 调整L_reg倍率
        else:
            loss = - entropy_loss + ce_loss + bce_loss # + 2 * L_reg1 + 2 * L_reg2 # 调整L_reg倍率
        
        print("index {}, entropy_loss {}, ce_loss {}, bce_loss {}, L_cluster {}, unlabel_ce_loss {}".format(batch_idx, entropy_loss.item(), ce_loss.item(), bce_loss.item(), L_cluster.item(), unlabel_ce_loss.item()))
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Final objective ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        
        # ------------------------------ Tracking ------------------------------
        if is_track:
            bce_losses.update(bce_loss.item(), args.batch_size)
            ce_losses.update(ce_loss.item(), args.batch_size)
            entropy_losses.update(entropy_loss.item(), args.batch_size)
        # ------------------------------ Tracking ------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if client_id == 0:
    # unlabel_acc, w_unlabel_acc = cluster_acc_w(np.array(cluster_pred[labeled_len:].cpu().numpy()), np.array(unlabel_target.cpu().numpy()))
    np_cluster_preds = np_cluster_preds.astype(int)
    unlabel_acc, w_unlabel_acc = cluster_acc_w(np_cluster_preds, np_unlabel_targets)
    print("unlabel_acc: ", unlabel_acc)
    print("w_unlabel_acc: ", w_unlabel_acc)
    # print("unlabel target: ", unlabel_target)
    # print("unlabel cluster_pred: ", cluster_pred[labeled_len:])
    #sys.exit(0)

    tf_writer.add_scalars('client{}/loss'.format(client_id), {"bce": bce_losses.avg}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"ce": ce_losses.avg}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"entropy": entropy_losses.avg}, args.epochs * global_round + epoch)

def test(args, model, labeled_num, device, test_loader, epoch, tf_writer, client_id, global_round):
    model.eval()
    preds = np.array([])
    cluster_preds = np.array([]) # cluster_preds
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        C = model.centroids.weight.data.detach().clone().T # C: [input, output] = [512, 10]
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1) # prediction of network
            # cluster pred
            Z1 = F.normalize(feat, dim=1)
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / model.T, dim=1) # prediciton from centroids
            _, cluster_pred = tP1.max(1) # return #1: max data    #2: max data index
            
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            cluster_preds = np.append(cluster_preds, cluster_pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    # Converts the targets array to an integer type.
    targets = targets.astype(int)
    preds = preds.astype(int)
    cluster_preds = cluster_preds.astype(int)

    seen_mask = targets < labeled_num # bool binary array, labeled class: True, unseen class: False
    unseen_mask = ~seen_mask # bool binary array, labeled class: False, unseen class: True

    ## local_unseen_mask (lu_mask) (6) ##
    local_unseen_mask_6 = targets == 6
    local_unseen_acc_6 = cluster_acc(preds[local_unseen_mask_6], targets[local_unseen_mask_6])
    ## gu_mask (7-9) (gu_mask)##
    gu_mask = targets > labeled_num
    gu_acc = cluster_acc(preds[gu_mask], targets[gu_mask])
    au_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    # unseen_acc, w_unseen_acc = cluster_acc_w(preds[unseen_mask], targets[unseen_mask])
    # if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
    #     print("w_unseen_acc: ", w_unseen_acc)

    overall_acc = cluster_acc(preds, targets)
    # overall_acc, w_overall_acc = cluster_acc_w(preds, targets)
    # if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
    #     print("w_overall_acc: ", w_overall_acc)

    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    
    # cluster_acc
    overall_cluster_acc = cluster_acc(cluster_preds, targets)
    
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)

    # Track
    print('epoch {}, Client id {}, Test overall acc {:.4f}, Test overall cluster acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, local_unseen acc {:.4f}, global_unseen acc {:.4f}'.format(epoch, client_id, overall_acc, overall_cluster_acc, seen_acc, au_acc, local_unseen_acc_6, gu_acc))
    # format(epoch, client_id, overall_acc, overall_cluster_acc, seen_acc, au_acc, local_unseen_acc_6, gu_acc)
    # 'add_scalar'
    # tf_writer.add_scalar('client{}/acc/overall'.format(client_id), overall_acc, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/acc/seen'.format(client_id), seen_acc, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/acc/unseen'.format(client_id), au_acc, args.epochs * global_round + epoch)
    # 'add_scalars'
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"overall": overall_acc}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"seen": seen_acc}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"unseen": au_acc}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_6": local_unseen_acc_6}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"global_unseen": gu_acc}, args.epochs * global_round + epoch)
    ##
    tf_writer.add_scalar('client{}/nmi/unseen'.format(client_id), unseen_nmi, args.epochs * global_round + epoch)
    tf_writer.add_scalar('client{}/uncert/test'.format(client_id), mean_uncert, args.epochs * global_round + epoch)
    return mean_uncert

def main():
    args, device = args_setting()

    # dataset initialization
    if args.dataset == 'cifar10':
        # train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), exist_label_list=[0,1,2,3,4,5,6,7,8,9], clients_num=args.clients_num)
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=10, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        # train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
        # test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
        num_classes = 10

        ### prepare clients dataset ###
        exist_label_list=[[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,9]]
        # Here 0,1,2,3,4,5 are the labeled classes for each client; 6 are global unseen; 7,8,9 are local unseen
        clients_labeled_num = [6, 6, 6, 6, 6] # the number of labeled class for each client (here number of clients equals 5)
        
        clients_train_label_set = []
        clients_train_unlabel_set = []
        clients_test_set = []
        for i in range(args.clients_num):
            client_train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=clients_labeled_num[i],
                                                        labeled_ratio=args.labeled_ratio, download=True,
                                                        transform=TransformTwice(
                                                            datasets.dict_transform['cifar_train']), exist_label_list=exist_label_list[i], clients_num=args.clients_num)
            client_train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False,
                                                          labeled_num=clients_labeled_num[i],
                                                          labeled_ratio=args.labeled_ratio, download=True,
                                                          transform=TransformTwice(
                                                              datasets.dict_transform['cifar_train']),
                                                          unlabeled_idxs=client_train_label_set.unlabeled_idxs, exist_label_list=exist_label_list[i], clients_num=args.clients_num)
            client_test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                        labeled_ratio=args.labeled_ratio, download=True,
                                                        transform=datasets.dict_transform['cifar_test'],
                                                        unlabeled_idxs=train_label_set.unlabeled_idxs,
                                                        exist_label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                        clients_num=args.clients_num)

            clients_train_label_set.append(client_train_label_set)
            clients_train_unlabel_set.append(client_train_unlabel_set)
            clients_test_set.append(client_test_set)

    # Not complete; pending to modify    
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100
    else:
        warnings.warn('Dataset is not listed')
        return

    # labeled batch size for each client is determined by the formula: labeled_batch_size = batch_size * labeled_len / (labeled_len + unlabeled_len)
    clients_labeled_batch_size = []
    for i in range(args.clients_num):
        labeled_len = len(clients_train_label_set[i])
        unlabeled_len = len(clients_train_unlabel_set[i])
        labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))
        clients_labeled_batch_size.append(labeled_batch_size)

    # Initialize the splits  # train_label_loader > client_train_label_loader[];   train_unlabel_loader -> client_train_unlabel_loader[]
    client_train_label_loader = []
    client_train_unlabel_loader = []
    client_test_loader = []
    for i in range(args.clients_num): # train_label_loader->client_train_label_loader[];   train_unlabel_loader -> client_train_unlabel_loader[]
        train_label_loader = torch.utils.data.DataLoader(clients_train_label_set[i], batch_size=clients_labeled_batch_size[i], shuffle=True, num_workers=16, drop_last=True)
        train_unlabel_loader = torch.utils.data.DataLoader(clients_train_unlabel_set[i], batch_size=args.batch_size - clients_labeled_batch_size[i], shuffle=True, num_workers=16, drop_last=True)
        client_train_label_loader.append(train_label_loader)
        client_train_unlabel_loader.append(train_unlabel_loader)
        # test_loader
        test_loader = torch.utils.data.DataLoader(clients_test_set[i], batch_size=100, shuffle=False, num_workers=1)
        client_test_loader.append(test_loader)

    # Initialize the global_model 
    global global_model
    global_model = models.resnet18(num_classes=num_classes) # for CIFAR model
    global_model = global_model.to(device)
    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    global_model.load_state_dict(state_dict, strict=False)
    global_model = global_model.to(device)


    # Freeze the earlier filters: similar to ORCA -> only 'layer4', 'linear' and 'centroids' are trainable
    for name, param in global_model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False
        if "centroids" in name:
            param.requires_grad = True

    # Initialize the clients' models, optimizers and schedulers
    clients_model = [] # model->clients_model[client_id]
    clients_optimizer = [] # optimizer->clients_optimizer[client_id]
    clients_scheduler = [] # scheduler->clients_scheduler[client_id]
    clients_tf_writer = [] # tf_writer->clients_tf_writer[client_id]
    for i in range(args.clients_num):
        model = copy.deepcopy(global_model)
        model = model.to(device)
        # Freeze the earlier filters of client models
        for name, param in model.named_parameters():
            if 'linear' not in name and 'layer4' not in name:
                param.requires_grad = False
            if "centroids" in name:
                param.requires_grad = False
        clients_model.append(model)

        # Set the optimizer
        optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        clients_optimizer.append(optimizer)
        clients_scheduler.append(scheduler)
        #tf_writer = SummaryWriter(log_dir=args.savedir)
        #clients_tf_writer.append(tf_writer)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    ## Start FedoAvg training ##
    for global_round in range(args.global_rounds):
        print("Start global_round {}: ".format(global_round))
        for client_id in range(args.clients_num):
            for epoch in range(args.epochs): # train local model in E epochs
                mean_uncert = test(args, clients_model[client_id], args.labeled_num, device, client_test_loader[client_id], epoch, tf_writer, client_id, global_round)
                train(args, clients_model[client_id], device, client_train_label_loader[client_id], client_train_unlabel_loader[client_id], clients_optimizer[client_id], mean_uncert, epoch, tf_writer, client_id, global_round)
                clients_scheduler[client_id].step()
            # local_clustering #
            clients_model[client_id].local_clustering(device=device)

        # receive_models/ upload local models
        receive_models(clients_model)
        # aggregate_parameters #global model average parameters across all the clients
        aggregate_parameters()

        # Run global clustering: concatenate `local centroids` parameters in each client into global variable `Z1`; `local centroids` [N_local, D]; 
        for client_id in range(args.clients_num):
            for c_name, old_param in clients_model[client_id].named_parameters(): # c_name: layer name; old_param: layer parameters
                if "local_centroids" in c_name:
                    if client_id == 0:
                        Z1 = np.array(copy.deepcopy(old_param.data.cpu().clone()))
                    else:
                        Z1 = np.concatenate((Z1, np.array(copy.deepcopy(old_param.data.cpu().clone()))), axis=0)
        Z1 = torch.tensor(Z1, device=device).T
        global_model.global_clustering(Z1.to(device).T) # update self.centroids in global model
        # set labeled data feature instead of self.centroids
        global_model.set_labeled_feature_centroids(device=device)

        # download global model param
        # name_filters = ['linear', "mem_projections", "centroids", "local_centroids"]
        # name_filters = ['linear', "mem_projections", "local_centroids", "local_labeled_centroids"] #do not AVG FedRep
        name_filters = ["mem_projections", "local_centroids", "local_labeled_centroids"]  # do not AVG FedAVG
        for client_id in range(args.clients_num):
            for (g_name, new_param), (c_name, old_param) in zip(global_model.named_parameters(), clients_model[client_id].named_parameters()):
                if all(keyword not in g_name for keyword in name_filters):
                    old_param.data = new_param.data.clone()
                    # print("Download layer name: ", g_name)
            # sys.exit(0)

    ## finish train ##        
                    
    # torch.save(global_model.state_dict(), './fedrep-trained-model/0102_Ours_cluster-classifier_global.pth')
    # for client_id in range(args.clients_num):
    #     torch.save(clients_model[client_id].state_dict(), './fedrep-trained-model/0102_Ours_cluster-classifier_client{}-model.pth'.format(client_id))
    ## save model

if __name__ == '__main__':
    main()
