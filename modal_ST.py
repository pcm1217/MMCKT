import numpy as np
import torch
from torch import nn
import math
from torch.nn import functional as F
import copy
#serimport Geohash



# prompt Aggregation Strategy 1
class Promptattention_a(nn.Module):
    def __init__(self,dk):
        super(Promptattention_a, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 
        return context, attn
    
class AttentionLayer_a(nn.Module):
    def __init__(self,args):
        super(AttentionLayer_a, self).__init__()
        self.args=args
        self.d_model=args.hidden_units
        self.dk=args.d_k
        self.n_heads=args.n_heads
        self.dv=args.d_v
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc = nn.Linear(self.d_model, self.d_model, bias=False)
    def forward(self, input):

        residual, batch_size = input, input.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input)  # Q: [batch_size, len, prompt, d_model]
        K = self.W_K(input) 
        V = self.W_V(input)  
        attn_shape = [input.size(0), input.size(2), input.size(2)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        subsequence_mask=subsequence_mask.unsqueeze(1).expand(-1, input.size(1), -1, -1).to(self.args.device)
 
        context, attn = Promptattention_a(self.dk)(Q, K, V,subsequence_mask)
        output = self.fc(context) 
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual)

class PromptLearner_a(nn.Module):
    def __init__(self, args,item_num):
        super().__init__()
        self.args=args
        
        emb_num = 2   
        emb_num_S =2  
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units) #share
        self.attention_E = AttentionLayer_a(args)
        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))
        ctx_vectors_E = embedding_E
        ctx_vectors_S_E = embedding_S_E
        self.ctx_E = nn.Parameter(ctx_vectors_E)  
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)


    def forward(self, seq ):
 
        seq_feat=self.src_emb(seq)
        positions = np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])
        seq_feat += self.pos_emb(torch.LongTensor(positions).to(self.args.device))
        seq_feat = self.emb_dropout(seq_feat)
        ctx_E = self.ctx_E 
        ctx_S_E = self.ctx_S_E 
        ctx_E_1 = ctx_E 

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  
   
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  

        ctx_prefix_E = self.getPrompts(seq_feat.unsqueeze(2), ctx_E, ctx_S_E ) # 128 15 8 100

        prompts_E = self.attention_E(ctx_prefix_E)[:,:,-1,:]
        return prompts_E

    def getPrompts(self, prefix, ctx,ctx_S): #ctx_S, suffix=None):#
    
        prompts = torch.cat(
            [
                ctx_S, 
                ctx,  
                prefix 
            ],
            dim=2,
        )
        return prompts


# prompt Aggregation Strategy 2

class Promptattention_b(nn.Module):
    def __init__(self,dk):
        super(Promptattention_b, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 
        return context, attn
    
class AttentionLayer_b(nn.Module):
    def __init__(self,args):
        super(AttentionLayer_b, self).__init__()
        self.args=args
        self.d_model=args.hidden_units
        self.dk=args.d_k
        self.n_heads=args.n_heads
        self.dv=args.d_v
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc = nn.Linear(self.d_model, self.d_model, bias=False)
    def forward(self, input):

        residual, batch_size = input, input.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input)  # Q: [batch_size, len, prompt, d_model]
        K = self.W_K(input) 
        V = self.W_V(input)  
        attn_shape = [input.size(0), input.size(2), input.size(2)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        subsequence_mask=subsequence_mask.unsqueeze(1).expand(-1, input.size(1), -1, -1).to(self.args.device)
 
        context, attn = Promptattention_b(self.dk)(Q, K, V,subsequence_mask)
        output = self.fc(context) 
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual)

class PromptLearner_b(nn.Module):
    def __init__(self, args,item_num):
        super().__init__()
        self.args=args
        
        emb_num = 2   
        emb_num_S =2  
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.max_len, args.hidden_units) # TO IMPROVE
        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units) #share
        self.attention_E = AttentionLayer_b(args)
        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))
        ctx_vectors_E = embedding_E
        ctx_vectors_S_E = embedding_S_E
        self.ctx_E = nn.Parameter(ctx_vectors_E)  
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)


    def forward(self, seq ):
 
        seq_feat=self.src_emb(seq)
        positions = np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])
        seq_feat += self.pos_emb(torch.LongTensor(positions).to(self.args.device))
        seq_feat = self.emb_dropout(seq_feat)
        ctx_E = self.ctx_E 
        ctx_S_E = self.ctx_S_E 
        ctx_E_1 = ctx_E 

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  
   
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  

        ctx_prefix_E = self.getPrompts(seq_feat.unsqueeze(2), ctx_E, ctx_S_E ) # 128 15 8 100
        number = ctx_prefix_E.shape[2]
        prompts_E = self.attention_E(ctx_prefix_E).sum(2)/number
        return prompts_E

    def getPrompts(self, prefix, ctx,ctx_S): 
    
        prompts = torch.cat(
            [
                ctx_S, 
                ctx,  
                prefix 
            ],
            dim=2,
        )
        return prompts

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
  
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   
    return pad_attn_mask.expand(batch_size, len_q, len_k) 
def get_attn_subsequence_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self,dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) 
        scores.masked_fill_(attn_mask, -1e9) 
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 

        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,args,d_model,d_k,n_heads,d_v):
        super(MultiHeadAttention, self).__init__()
        self.args=args
        self.d_model=d_model
        self.dk=d_k
        self.n_heads=n_heads
        self.dv=d_v
        self.W_Q = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.dv * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dv, self.d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dv).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dk)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,(self.n_heads) * (self.dv)) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args=args
        self.d_model=d_model
        self.d_ff=d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
    def forward(self, inputs):

        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):

    def __init__(self,args,d_model,d_k,n_heads,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args,d_model,d_k,n_heads,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(args,d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )
    def forward(self, x, mask=None):
    
        out = self.linear(x)
        if mask is not None:  
            out = out.masked_fill(mask, -100000)  
            weight = F.softmax(out, dim=1)
            return weight
        else:
            weight = F.softmax(out, dim=2) 
        return weight 


class ContrastiveLearningModule(nn.Module):
    def __init__(self, hidden_dim, temperature=0.1):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, *modalities):
        projected_modalities = [self.projection_head(modality).mean(dim=1) for modality in modalities]
        normalized_modalities = [F.normalize(proj_modality, p=2, dim=1) for proj_modality in projected_modalities]

        logits = torch.matmul(normalized_modalities[0], normalized_modalities[1].t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        return contrastive_loss

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(15, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(15, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x



class PromptLearner(nn.Module):
    def __init__(self, args, item_num, modality_size):
        super().__init__()
        self.args = args
        emb_num = 2
        emb_num_S = 2
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.src_emb = nn.Embedding(item_num, args.hidden_units)

        #self.loc_emb = nn.Embedding(item_num, args.hidden_units)

        self.time_emb = Time2Vec('sin', out_dim=args.hidden_units)

        self.pos_emb = torch.nn.Embedding(args.max_len, args.hidden_units)

        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units * 5)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units * 5)
        #self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units * 4)
        #self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units * 4)
        #self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
        #self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units)

        drop_out = 0.25

        self.attention_E = AttentionLayer(10 * args.hidden_units, drop_out)
        #self.attention_E = AttentionLayer(8 * args.hidden_units, drop_out)
        #self.attention_E = AttentionLayer(2 * args.hidden_units, drop_out)


        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))

        ctx_vectors_E = embedding_E
        ctx_vectors_S_E = embedding_S_E

        self.ctx_E = nn.Parameter(ctx_vectors_E)
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)

        self.modality_linear = nn.Linear(args.multimodal_dim, args.hidden_units)

        self.contrastive_module = ContrastiveLearningModule(args.hidden_units)

        self.eps = 0.5


    def forward(self, seq,  time_seq, img_emb, text_emb, meta_emb):
        #print(f"seq.shape: {seq.shape}")
        #print(f"seq.shape[1]: {seq.shape[1]}")
        seq_feat = self.src_emb(seq)
        positions = torch.arange(seq.shape[1]).expand(seq.shape[0], seq.shape[1]).to(self.args.device)
        seq_feat += self.pos_emb(positions)
        seq_feat = self.emb_dropout(seq_feat)


         # Embedding location sequence
        #loc_feat = self.loc_emb(loc_seq)

        time_emb = []
        loc_emb = []
        # Embedding time sequence
        for idx in range(len(seq)):
            time_feat = self.time_emb(time_seq[idx])
            time_emb.append(time_feat)
            #loc_feat = self.loc_emb(loc_seq[idx])
            #loc_emb.append(loc_feat)
        time_emb = torch.stack(time_emb)
        #loc_emb = torch.stack(loc_emb)

        # Enhance multimodal features
        img_emb = self.modality_linear(img_emb)
        text_emb = self.modality_linear(text_emb)
        meta_emb = self.modality_linear(meta_emb)

        random_noise = torch.rand_like(img_emb, device=img_emb.device)
        img_emb1 = img_emb + torch.sign(img_emb) * F.normalize(random_noise, dim=-1) * self.eps

        random_noise = torch.rand_like(img_emb, device=img_emb.device)
        img_emb2 = img_emb + torch.sign(img_emb) * F.normalize(random_noise, dim=-1) * self.eps

        random_noise = torch.rand_like(text_emb, device=text_emb.device)
        text_emb1 = text_emb + torch.sign(text_emb) * F.normalize(random_noise, dim=-1) * self.eps

        random_noise = torch.rand_like(text_emb, device=text_emb.device)
        text_emb2 = text_emb + torch.sign(text_emb) * F.normalize(random_noise, dim=-1) * self.eps

        random_noise = torch.rand_like(meta_emb, device=meta_emb.device)
        meta_emb1 = meta_emb + torch.sign(meta_emb) * F.normalize(random_noise, dim=-1) * self.eps

        random_noise = torch.rand_like(meta_emb, device=meta_emb.device)
        meta_emb2 = meta_emb + torch.sign(meta_emb) * F.normalize(random_noise, dim=-1) * self.eps

        #contrastive_loss = self.contrastive_module(img_emb, text_emb, meta_emb)
        contrastive_loss = 0
        combined_emb = torch.cat((seq_feat,  time_emb, img_emb, text_emb, meta_emb), dim=-1)
        #combined_emb = torch.cat((seq_feat, img_emb, text_emb, meta_emb), dim=-1)
        #combined_emb = seq_feat + time_emb + img_emb + text_emb + meta_emb
        #combined_emb = seq_feat

        ctx_E = self.ctx_E
        ctx_S_E = self.ctx_S_E
        ctx_E_1 = ctx_E

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1], -1, -1)
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1], -1, -1)

        ctx_prefix_E = self.getPrompts(combined_emb.unsqueeze(2), ctx_E, ctx_S_E)

        item_embedding = combined_emb.unsqueeze(2).expand(-1, -1, ctx_prefix_E.shape[2], -1)

        prompt_item = torch.cat((ctx_prefix_E, item_embedding), dim=3)
        at_wt = self.attention_E(prompt_item)
        prompts_E = torch.matmul(at_wt.permute(0, 1, 3, 2), ctx_prefix_E).squeeze()

        return prompts_E, contrastive_loss, img_emb1, img_emb2, text_emb1, text_emb2, meta_emb1, meta_emb2

    def getPrompts(self, prefix, ctx, ctx_S):
        prompts = torch.cat([ctx, ctx_S, prefix], dim=2)
        return prompts


import torch.nn.functional as F


class mcrpl(nn.Module):
    def __init__(self,args,item_num):
        super(mcrpl, self).__init__()
        self.args=args
        #self.class_=nn.Linear(args.hidden_units,args.all_size)
        self.class_=nn.Linear(args.hidden_units*5,args.all_size)
        #self.class_=nn.Linear(args.hidden_units*4,args.all_size)
        # self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        # self.pos_emb = PositionalEncoding(args.hidden_units)
        # 定义图像嵌入的MLP
        #self.layers = nn.ModuleList([EncoderLayer(self.args,args.hidden_units,args.d_k,args.n_heads,args.d_v,args.d_ff) for _ in range(args.n_layers)])
        #self.layers = nn.ModuleList([EncoderLayer(self.args,args.hidden_units*4,args.d_k,args.n_heads,args.d_v,args.d_ff) for _ in range(args.n_layers)])
        self.layers = nn.ModuleList([EncoderLayer(self.args,args.hidden_units*5,args.d_k,args.n_heads,args.d_v,args.d_ff) for _ in range(args.n_layers)])
        if args.Strategy == 'default' :
            self.prompt=PromptLearner(args,item_num, 3)
        elif args.Strategy == 'a':
            self.prompt=PromptLearner_a(args,item_num, 3)
        elif args.Strategy == 'b':
            self.prompt=PromptLearner_b(args,item_num, 3)

    def phase_one(self, user, log_seqs, time_seq, img_emb, text_emb, meta_emb):
        
        
        enc_outputs, contrastive_loss, img_emb1, img_emb2, text_emb1, text_emb2, meta_emb1, meta_emb2 = self.prompt(log_seqs, time_seq, img_emb, text_emb, meta_emb)

        enc_self_attn_mask = get_attn_pad_mask(log_seqs, log_seqs) # [batch_size, src_len, src_len]

        enc_attn_mask=get_attn_subsequence_mask(log_seqs).to(self.args.device)

        all_mask=torch.gt((enc_self_attn_mask + enc_attn_mask), 0).to(self.args.device)
        
        enc_self_attns = []
        for layer in self.layers:
            
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
    
            enc_self_attns.append(enc_self_attn)
   
        logits=self.class_(enc_outputs[:,-1,:])
        return logits, contrastive_loss, img_emb1, img_emb2, text_emb1, text_emb2, meta_emb1, meta_emb2, self.prompt.ctx_E,  self.prompt.ctx_S_E
    

    def forward(self,user,log_seqs, time_seq, img_emb, text_emb, meta_emb):
        logits=self.phase_one(user,log_seqs, time_seq, img_emb, text_emb, meta_emb) #log_seqs:[batch_size, src_len] img_emb:[batch_size,src_len, img_dim]
        return logits
    
    def Freeze_a(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():

            if "ctx_S_E" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = False
        self.prompt.pos_emb.requires_grad = False
        self.prompt.time_emb.requires_grad = False
        

    def Freeze_b(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            #print(name)
            if "ctx" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True

    def Freeze_c(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if "layers" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True
        
    def Freeze_d(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if "ctx" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True