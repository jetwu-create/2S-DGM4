import torch
import torch.nn as nn
from modules.Transformer import  MELDTransEncoder, AdditiveAttention
from modules.CrossmodalTransformer import CrossModalTransformerEncoder
from transformers import RobertaModel
from transformers import BertModel
from modules.SwinTransformer.backbone_def import BackboneFactory
import torch.nn.functional as F
import pdb

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


'''SwinTransformer_modify'''
class SwinForAffwildClassification(nn.Module):

    def __init__(self, args):
        super(SwinForAffwildClassification,self).__init__()
        # self.num_labels = 7
        self.swin = BackboneFactory(args.backbone_type, args.backbone_conf_file).get_backbone() #加载Swin-tiny模型
        # Classifier head
        self.linear = nn.Linear(768, 768)
        self.nonlinear = nn.ReLU()
        # self.classifier = nn.Linear(64, 7)
        # self.tau = 1  #temperature parameter

    def forward(self, images_feature=None):
        # pdb.set_trace()
        outputs = self.swin(images_feature)
        outputs = self.linear(outputs)
        outputs = self.nonlinear(outputs)
        return outputs
        # logits = self.classifier(outputs) 
        # if is_trg_task:
        #     logits = F.gumbel_softmax(logits, self.tau)  
        # if labels is not None:
        #     loss = criterion(logits, labels) #cross entropy loss
        #     return loss
        # else:
        #     return logits



'''SwinTransformer_orig'''
class SwinForAffwildClassification_orig(nn.Module):

    def __init__(self, args):
        super(SwinForAffwildClassification_orig,self).__init__()
        # pdb.set_trace()
        self.num_labels = args.num_labels
        self.swin = BackboneFactory(args.backbone_type, args.backbone_conf_file).get_backbone() #加载Swin-tiny模型
        # Classifier head
        self.linear = nn.Linear(512, 64)
        self.nonlinear = nn.ReLU()
        self.classifier = nn.Linear(64, args.num_labels)
        self.tau = args.tau  #temperature parameter

    def forward(self, images_feature=None, is_trg_task=None, labels=None, criterion=None):
        # pdb.set_trace()
        outputs = self.swin(images_feature) 
        # pdb.set_trace()
        outputs = self.linear(outputs)
        outputs = self.nonlinear(outputs)
        # pdb.set_trace()
        logits = self.classifier(outputs) 
        if is_trg_task:
            logits = F.gumbel_softmax(logits, self.tau)  
        if labels is not None:
            loss = criterion(logits, labels) #cross entropy loss
            return loss
        else:
            return logits