import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import copy
import numpy as np
import math

from eval import segment_bars_with_confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)

        # print('proj_query shape ', proj_query.shape)
        # print('proj_key shape ', proj_key.shape)
        # print('proj_val shape ', proj_val.shape)
        print('attention shape ', attention.shape)
        # print('padding_mask shape ', padding_mask.shape, padding_mask[0])
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        print('out and attention: ', out.shape, attention.shape)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)
        # self.conv_out = nn.Conv1d(in_channels=v_dim, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, :, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        print(x1.shape)
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
            # key = self.key_conv(x2)
        else: 
            value = self.value_conv(x1)
            # key = self.key_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    
    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
    
    def _sliding_window_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        # assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl 
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) 
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        # print('before padding_mask ', padding_mask)
        # print('before window_mask ', self.window_mask)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask 

        print('-->>>', q.shape, k.shape, v.shape)
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]
    


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    
class AttModuleDDL(nn.Module):
    def __init__(self, layer_lvl, num_layers, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModuleDDL, self).__init__()

        dilation1, dilation2 = 2 ** layer_lvl, 2 ** (num_layers - 1 - layer_lvl)
        #window_size = max(dilation1, 32)
        window_size = dilation1
        print('lvl: ', layer_lvl, ' d1: ', dilation1, ' d2: ', dilation2, ' window_size: ', window_size, flush=True)

        self.feed_forward_1 = ConvFeedForward(dilation1, in_channels, out_channels)
        self.feed_forward_2 = ConvFeedForward(dilation2, in_channels, out_channels)
        self.conv_fusion = nn.Conv1d(2 * out_channels, out_channels, 1)

        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, window_size, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        
        out = self.conv_fusion(torch.cat([self.feed_forward_1(x), self.feed_forward_2(x)], 1))
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return (x + out) * mask[:, 0:1, :]


class AttModuleDDA(nn.Module):
    def __init__(self, layer_lvl, num_layers, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModuleDDA, self).__init__()

        dilation1, dilation2 = 2 ** layer_lvl, 2 ** (num_layers - 1 - layer_lvl)
        print('lvl: ', layer_lvl, ' d1: ', dilation1, ' d2: ', dilation2, flush=True)

        self.feed_forward_1 = ConvFeedForward(dilation1, in_channels, out_channels)
        self.feed_forward_2 = ConvFeedForward(dilation2, in_channels, out_channels)

        self.instance_norm_1 = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.instance_norm_2 = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer_1 = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation1, att_type=att_type, stage=stage) # dilation
        self.att_layer_2 = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation2, att_type=att_type, stage=stage)
        self.conv_fusion = nn.Conv1d(2 * out_channels, out_channels, 1)
        #self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        
        out1 = self.feed_forward_1(x)
        out1 = self.alpha * self.att_layer_1(self.instance_norm_1(out1), f, mask) + out1
        
        out2 = self.feed_forward_2(x)
        out2 = self.alpha * self.att_layer_2(self.instance_norm_2(out2), f, mask) + out2
        
        out = self.conv_fusion(torch.cat([out1, out2], 1))
        #out = self.conv_1x1(out)
        out = self.dropout(out)
        
        return (x + out) * mask[:, 0:1, :]


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()

        dilation = min(dilation, 2048)
        print(' d: ', dilation, flush=True)
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class FixedPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 10000):
        
        super(FixedPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)  #(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) #(div_term) 

        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term) #(max_len, div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :] # (1, T, 2048)

        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(LearnablePositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:, :x.size(1), :] # (1, T, 2048)

        return self.dropout(x)

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha, arch_type, pos_encoding):
        super(Encoder, self).__init__()
        
        self.conn = arch_type[1] if len(arch_type) > 1 else None 
        print('*** Encoder: ', self.conn, ' ***')
        self.num_layers = num_layers
        self.is_pos_enc = 1 if pos_encoding is not None else 0

        if self.is_pos_enc:
            print('Encoder pos_encoding: ', pos_encoding, flush=True)
            self.position_en = get_pos_encoder(pos_encoding)(num_f_maps, dropout=0.1)

        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer

        if arch_type[0] == 'ddl':
            print('Encoder arch: ddl - ', arch_type, flush=True)
            self.layers = nn.ModuleList(
            [AttModuleDDL(i, num_layers, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in
             range(num_layers)])

        elif arch_type[0] == 'dda':
            print('Encoder arch: dda - ', arch_type, flush=True)
            self.layers = nn.ModuleList(
            [AttModuleDDA(i, num_layers, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in
             range(num_layers)])

        else:
            print('Encoder arch: default - ',arch_type, flush=True)
            self.layers = nn.ModuleList(
                [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
                range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        # x shape : (B, 2048, L)

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)

        # add pos encoding
        if self.is_pos_enc:
            feature = feature.permute(0, 2, 1) # (B, C, L) -> (B, L, C)
            feature = self.position_en(feature)
            feature = feature.permute(0, 2, 1) # (B, L, C) -> (B, C, L)
        
        bs, num_f_maps, L = feature.size()

        if self.conn:
            enc_features = torch.empty(self.num_layers, bs, num_f_maps, L).to(device) 
        
        for num_layer, layer in enumerate(self.layers):
            feature = layer(feature, None, mask)
            if self.conn:  # for skip connection
                enc_features[num_layer] = feature.reshape(1, bs, num_f_maps, L)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]
    
        return (out, enc_features) if self.conn else (out, feature) 


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha, arch_type, pos_encoding, num_dec):
        super(Decoder, self).__init__()         
        
        self.conn = arch_type[1] if len(arch_type) > 1 else None
        print('*** Decoder: ', self.conn, ' ***')
        self.num_layers = num_layers 
        self.num_dec = num_dec
        self.is_pos_enc = 1 if pos_encoding is not None else 0

        if self.is_pos_enc and self.num_dec == 0:
            print('Decoder pos_encoding: ', pos_encoding, flush=True)
            self.position_en = get_pos_encoder(pos_encoding)(num_f_maps, dropout=0.1)

        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)

        if arch_type[0] == 'ddl':
            self.layers = nn.ModuleList(
                [AttModuleDDL(i, num_layers, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in
                range(num_layers)])

        elif arch_type[0] == 'dda':
            self.layers = nn.ModuleList(
                [AttModuleDDA(i, num_layers, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in
                range(num_layers)])

        else:
            self.layers = nn.ModuleList(
                [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
                range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)

        if self.is_pos_enc and self.num_dec == 0:
            # add pos encoding
            feature = feature.permute(0, 2, 1) # (B, C, L) -> (B, L, C)
            feature = self.position_en(feature)
            feature = feature.permute(0, 2, 1) # (B, L, C) -> (B, C, L)

        bs, num_f_maps, L = feature.size()

        if self.conn:
            dec_features = torch.empty(self.num_layers, bs, num_f_maps, L).to(device) 

        for num_layer, layer in enumerate(self.layers):
            if self.conn == 'ca_enc': 
                #feature = layer(feature, fencoder[-num_layer - 1], mask) # unet order
                feature = layer(feature, fencoder[num_layer], mask) # cet order
            elif self.conn =='ca':
                #feature = layer(feature, fencoder[-num_layer - 1], mask) # unet order
                feature = layer(feature, fencoder[num_layer], mask) # cet order
                dec_features[num_layer] = feature.reshape(1, bs, num_f_maps, L)
            else:
                feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]
        
        return (out, dec_features) if self.conn else (out, feature) 
    
class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, arch_type, pos_enc):
        super(MyTransformer, self).__init__()
        arch_type =  arch_type.split('_', 1)
        self.conn = arch_type[1] if len(arch_type) > 1 else None 
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att', alpha=1, arch_type=arch_type, pos_encoding=pos_enc)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', arch_type=arch_type, alpha=exponential_descrease(s), pos_encoding=pos_enc, num_dec=s)) for s in range(num_decoders)]) # num_decoders
        
    def forward(self, x, mask):
        out, enc_feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            if self.conn == 'ca_enc': 
                out, _ = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], enc_feature * mask[:, 0:1, :], mask)
            elif self.conn == 'ca':
                out, enc_feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], enc_feature * mask[:, 0:1, :], mask)
            else:
                out, enc_feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], enc_feature * mask[:, 0:1, :], mask)
            
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
 
        return outputs

class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, arch_type, pos_enc):
        self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, arch_type, pos_enc)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()), flush=True)
        self.mse = nn.MSELoss(reduction ='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('LR:{}, BS:{}'.format(learning_rate, batch_size), flush=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        best_acc = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            start_time = time.time()
            while batch_gen.has_next():
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                ps = self.model(batch_input, mask) #(4, 12, 48, 832)

                loss = 0
                for p in ps:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=192) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(ps.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
                        
            scheduler.step(epoch_loss)
            batch_gen.reset()

            end_time = time.time()
            time_elapsed = (end_time - start_time) / 60 #mins

            print("Epoch: {}, Time: {:.4f}, Train loss: {:.4f}, Train acc: {:.4f}".format(
		epoch + 1, time_elapsed, epoch_loss / len(batch_gen.list_of_examples), float(correct) / total ), flush=True)

            if (epoch + 1) % 10 == 0 and batch_gen_tst is not None:
                acc = self.test(batch_gen_tst, epoch)
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
                
                if acc > best_acc:
                    print("#### Best Model Saved ####")
                    best_acc = acc
                    torch.save(self.model.state_dict(), save_dir + "/best_model.model")
                    torch.save(optimizer.state_dict(), save_dir + "/best_model.opt")


    def test(self, batch_gen_tst, epoch):
        self.model.eval()
        correct = 0
        total = 0
        if_warp = False  # When testing, always false
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, if_warp)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                p = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

        acc = float(correct) / total
        print("---[epoch %d]---: tst acc = %f" % (epoch + 1, acc), flush=True)

        self.model.train()
        batch_gen_tst.reset()
        return acc

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            # self.model.load_state_dict(torch.load(model_dir + "/best_model.model"))

            batch_gen_tst.reset()
            import time
            
            time_start = time.time()
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0]
#                 print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                if features.shape[1] == 768 or features.shape[1] == 894 or features.shape[1] == 512:
                    features = np.swapaxes(features,0,1)

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    if 0:
                        segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                    confidence.tolist(),
                                                    batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            time_end = time.time()
            
            

if __name__ == '__main__':
    pass
