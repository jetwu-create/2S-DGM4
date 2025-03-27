from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
from src.models import SwinForAffwildClassification
from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_
import pdb

class HAMMER(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 text_encoder = None,
                 tokenizer = None,
                 init_deit = True
                 ):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.load("/data/jetwu/plm/deit_base_patch16_224-b5f2ef4d.pth", map_location='cpu')
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForTokenClassification.from_pretrained(text_encoder, 
                                                                    config=bert_config, 
                                                                    label_smoothing=config['label_smoothing'])      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        # swin_transformer
        self.shareSwin_model = SwinForAffwildClassification(args)
        '''loading the pretrained SWIN (Swin Transformer) model and updating the parameter dictionary.'''
        if init_deit:
                model_dict = self.shareSwin_model.state_dict()
                pretrained_dict = torch.load(args.pretrained_backbone_path, map_location="cpu")['state_dict']
                new_pretrained_dict = {}
                for k in model_dict:
                    if k in pretrained_dict:
                        if k == 'classifier.weight':
                            continue
                        if k == 'classifier.bias':
                            continue
                        if k[:5] == 'swin.':
                            k_val = k[5:]
                        else:
                            k_val = k
                        new_pretrained_dict[k] = pretrained_dict['backbone.' + k_val] # tradition training
                model_dict.update(new_pretrained_dict)
                self.shareSwin_model.load_state_dict(model_dict)

        # creat itm head
        self.itm_head_F = self.build_mlp(input_dim=text_width, output_dim=3)
        self.itm_head_T = self.build_mlp(input_dim=text_width, output_dim=3)
        self.itm_head_C = self.build_mlp(input_dim=text_width, output_dim=5)

        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # creat multi-cls head
        # self.cls_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForTokenClassification.from_pretrained(text_encoder, 
                                                                    config=bert_config,
                                                                    label_smoothing=config['label_smoothing'])       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr =nn.LayerNorm(text_width)
        # self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        # trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward(self, image, label, text, fake_image_box, fake_text_pos, alpha=0, is_train=True):
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)
            ##================= multi-label convert ========================## 
            real_label_pos, FS_pos, FA_pos, TS_pos, TA_pos, FSTS_pos, FSTA_pos, FATS_pos, FATA_pos = get_multi_label(label, image)
            
            ##================= MAC ========================## 
            image_embeds = self.visual_encoder(image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
                
            # get momentum features
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image) 
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)
                # jinyu: local features of visual part
                image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m[:,1:,:]),dim=-1)
                image_feat_m_l = self.patch_pooling(image_feat_m_l) # pooling for image patches
                image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)

                text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                    return_dict = True, mode = 'text')
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
                # jinyu: local features of text part
                text_feat_m_l = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,1:,:]),dim=-1) 
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                # fine-grained alignment: only orig should be aligned, 1 here means img-text aligned 
                sim_targets[real_label_pos, real_label_pos] = 1 

                sim_targets_g2g = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets_g2g.fill_diagonal_(1)       
                
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

            sim_i2t = image_feat @ text_feat_all / self.temp 
            sim_t2i = text_feat @ image_feat_all / self.temp 
                                
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

            # jinyu: add inMod g2l loss
            loss_t2t_inMod_l = self.in_batch_g2l_loss(text_feat_m_l, text_feat, self.temp, text.attention_mask[:,1:])
            loss_i2i_inMod_l = self.in_batch_g2l_loss(image_feat_m_l, image_feat, self.temp)
            
            # in-modality g2g loss
            sim_i2i = image_feat @ image_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_g2g,dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_g2g,dim=1).mean()

            loss_MAC = (loss_i2t+loss_t2i+loss_i2i+loss_t2t+loss_t2t_inMod_l+loss_i2i_inMod_l)/6

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

            ##================= BIC ========================## 
            # forward the positve image-text pair
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )            
            with torch.no_grad():
                bs = image.size(0)          

            #Face
            itm_labels_F = torch.zeros(bs, dtype=torch.long).to(image.device)
            itm_labels_F[FS_pos] = 1
            itm_labels_F[FA_pos] = 2
            output_cls_F = self.itm_head_F(output_pos.last_hidden_state[:,0,:])
            loss_BIC_F = F.cross_entropy(output_cls_F, itm_labels_F)
            
            #Text
            itm_labels_T = torch.zeros(bs, dtype=torch.long).to(image.device)
            itm_labels_T[TS_pos] = 1
            itm_labels_T[TA_pos] = 2
            output_cls_T = self.itm_head_T(output_pos.last_hidden_state[:,0,:])
            loss_BIC_T = F.cross_entropy(output_cls_T, itm_labels_T)
            
            #Inte
            itm_labels_C = torch.zeros(bs, dtype=torch.long).to(image.device)
            itm_labels_C[FSTS_pos] = 1
            itm_labels_C[FSTA_pos] = 2
            itm_labels_C[FATS_pos] = 3
            itm_labels_C[FATA_pos] = 4
            output_cls_C = self.itm_head_C(output_pos.last_hidden_state[:,0,:])
            loss_BIC_C = F.cross_entropy(output_cls_C, itm_labels_C)
            
            loss_BIC = loss_BIC_F + loss_BIC_T + loss_BIC_C
            
            cls_tokens_local = self.shareSwin_model(image).unsqueeze(1)

            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            local_feat_it_cross_attn = image_embeds + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(text_embeds), 
                                              value=self.norm_layer_it_cross_atten(text_embeds),
                                              key_padding_mask=local_feat_padding_mask_text)[0]

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
            
            ##================= TMG ========================##    
            token_label = text.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
            token_label[token_label==0] = -100 # -100 index = padding token
            token_label[token_label==1] = 0

            for batch_idx in range(len(fake_text_pos)):
                fake_pos_sample = fake_text_pos[batch_idx]
                if fake_pos_sample:
                    for pos in fake_pos_sample:
                        token_label[batch_idx, pos] = 1

            input_ids = text.input_ids.clone()

            if self.args.token_momentum:
                with torch.no_grad():
                    logits_m = self.text_encoder_m(input_ids, 
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds_m,
                                                encoder_attention_mask = image_atts,      
                                                return_dict = True,
                                                return_logits = True,   
                                                )    
                token_cls_output = self.text_encoder(input_ids, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            labels = token_label,   
                                            soft_labels = F.softmax(logits_m.view(-1, 2),dim=-1),
                                            alpha = alpha
                                            )    
            else:
                token_cls_output  = self.text_encoder(input_ids, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            labels = token_label,   
                                            )  

            loss_TMG = token_cls_output.loss

            return loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG

        else:
            image_embeds = self.visual_encoder(image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state

            # forward the positve image-text pair
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )               
            ##================= IMG ========================## 
            bs = image.size(0)
            # cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)
            cls_tokens_local = self.shareSwin_model(image).unsqueeze(1)

            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            local_feat_it_cross_attn = image_embeds + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(text_embeds), 
                                              value=self.norm_layer_it_cross_atten(text_embeds),
                                              key_padding_mask=local_feat_padding_mask_text)[0]

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            ##================= OUR ========================##
            logits_fake_Face = self.itm_head_F(output_pos.last_hidden_state[:,0,:])
            logits_fake_Text = self.itm_head_T(output_pos.last_hidden_state[:,0,:])
            logits_fake_Com = self.itm_head_C(output_pos.last_hidden_state[:,0,:])
            ##================= TMG ========================##   
            input_ids = text.input_ids.clone()
            logits_tok = self.text_encoder(input_ids, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        return_logits = True,   
                                        )     
            return logits_fake_Face, logits_fake_Text, logits_fake_Com, output_coord, logits_tok   


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 


    # jinyu: patch pooling of image patches to reduce computation and enlarge receptive field
    def patch_pooling(self, x):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, c1*c1, dim)
        return x


    # jinyu: in-batch g2l loss
    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim) # (N * n_locals) * d
        m_n = m.reshape(-1, dim) # N * d

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp # N * n_locals * 1 * 1
        
        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1-temp_mask))
        
        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device) # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1-temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

