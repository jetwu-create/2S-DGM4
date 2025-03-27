import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
# from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label, our_multi_modify
from models.HAMMER import HAMMER


def zero_in_list(lst):
    return any(x == 0 for x in lst)

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)
        
        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)
        
        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch

  

@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    print_freq = 200 

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
    
    TP_all_multicls = np.zeros(4, dtype = int)
    TN_all_multicls = np.zeros(4, dtype = int)
    FP_all_multicls = np.zeros(4, dtype = int)
    FN_all_multicls = np.zeros(4, dtype = int)
    F1_multicls = np.zeros(4)

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)

        logits_fake_Face, logits_fake_Text, logits_fake_Com, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)

        ##================= real/fake cls ========================## 
        # cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        # real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        # cls_label[real_label_pos] = 0

        # pred_acc = logits_real_fake.argmax(1)
        # cls_nums_all += cls_label.shape[0]
        
        # ----- multiple labels -----
        # target, _ = get_multi_label(label, image)
        
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        IOU_pred.extend(IOU.cpu().tolist())

        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU>0.5] = 1
        IOU_75_bt[IOU>0.75] = 1
        IOU_95_bt[IOU>0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
        token_label[token_label==0] = -100 # -100 index = padding token
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1
                    
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)

        # #our modify evaluation
        # output_coord_convert = box_ops.box_cxcywh_convert(output_coord, W, H)               #image grounding coordinate[x1,y1,x2,y2]
        # output_coord_convert = np.array(output_coord_convert.cpu(), dtype=int).tolist()
        # bs_ = text_input.attention_mask[:,1:].shape[0]
        # text_seq_len = text_input.attention_mask[:,1:].shape[1]
        # logits_tok_pred_split = logits_tok_pred.view(bs_, text_seq_len)
        # token_label_reshape_split = token_label_reshape.view(bs_, text_seq_len)
        
        # # image_grounding+text_grounding评价real/fake
        # for i, item in enumerate(token_label_reshape_split):
        #     item = np.array(item.cpu())
        #     if np.where(item==-100)[0] != []:
        #         indices = np.where(item==-100)[0][0]
        #         text_grounding = np.array(logits_tok_pred_split[i][:indices].cpu())
        #     else:
        #         text_grounding = np.array(logits_tok_pred_split[i].cpu())
        #     image_groudning = output_coord_convert[i]
        #     if zero_in_list(text_grounding) and zero_in_list(image_groudning):
        #         pred_acc[i] = 0
        #     else:
        #         pred_acc[i] = 1
        # cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # F1
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()
                 
    ##================= real/fake cls ========================## 
    # ACC_cls = cls_acc_all / cls_nums_all
    ##================= bbox cls ========================##
    IOU_score = sum(IOU_pred)/len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50)/len(IOU_50)
    IOU_ACC_75 = sum(IOU_75)/len(IOU_75)
    IOU_ACC_95 = sum(IOU_95)/len(IOU_95)
    # ##================= token cls========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all)
    Recall_tok = TP_all / (TP_all + FN_all)
    F1_tok = 2*Precision_tok*Recall_tok / (Precision_tok + Recall_tok)
    ##================= multi-label cls ========================##
    # OP, OR, OF1, CP, CR, CF1 = our_multi_modify(, target)
    
    return IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'
    log_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')
    logger = setlogger(log_file)
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    #### Model #### 
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    if args.log:
        print(f"Creating MAMMER")
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    
    model = model.to(device)   

    checkpoint_dir = f'{args.output_dir}/{args.log_num}/checkpoint_{args.test_epoch}.pth'
    checkpoint = torch.load(checkpoint_dir, map_location='cpu') 
    state_dict = checkpoint['model']                       

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
                   
    # model.load_state_dict(state_dict)  
    if args.log:
        print('load checkpoint from %s'%checkpoint_dir)  
    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(msg)  

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    _, val_dataset = create_dataset(config)
    
    if args.distributed:  
        samplers = create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None]

    val_loader = create_loader([val_dataset],
                                samplers,
                                batch_size=[config['batch_size_val']], 
                                num_workers=[4], 
                                is_trains=[False], 
                                collate_fns=[None])[0]

    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.log:
        print("Start evaluation")


    IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
    ACC_tok, Precision_tok, Recall_tok, F1_tok  = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)
    #============ evaluation info ============#
    val_stats = {
                    "IOU_score": "{:.4f}".format(IOU_score*100),
                    "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                    "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                    "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                    "ACC_tok": "{:.4f}".format(ACC_tok*100),
                    "Precision_tok": "{:.4f}".format(Precision_tok*100),
                    "Recall_tok": "{:.4f}".format(Recall_tok*100),
                    "F1_tok": "{:.4f}".format(F1_tok*100),
    }
    
    if utils.is_main_process(): 
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': args.test_epoch,
                    }             
        with open(os.path.join(log_dir, f"results_{eval_type}.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='/mnt/lustre/share/rshao/data/FakeNews/Ours/results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--test_epoch', default='best', type=str)
    # parser.add_argument('--test_epoch', default='best', type=str)


    '''Swin Transformer backbone loading'''
    parser.add_argument("--backbone_type", type=str, default='SwinTransformer')
    parser.add_argument("--backbone_conf_file", type = str, default= './modules/SwinTransformer/swin_conf.yaml', help = "The path of backbone_conf.yaml.")
    parser.add_argument('--pretrained_backbone_path', type=str, default='Swin_tiny_Ms-Celeb-1M.pt')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
 
    main_worker(0, args, config)
