import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from deap_model.layers import general_conv3d_prenorm, fusion_prenorm

basic_dims = 16
transformer_basic_dims = 128
mlp_dim = 16
num_heads = 8
depth = 1
num_modals = 2
patch_size = 4

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)   ### B *16*32*32-->B*32*16*16
        x1 = x1 + self.e1_c3(self.e1_c2(x1))    ###B*32*16*16

        x2 = self.e2_c1(x1)   ### B*32*16*16 --> B*64*8*8
        x2 = x2 + self.e2_c3(self.e2_c2(x2))   ###B*64*8*8
 
        x3 = self.e3_c1(x2)   ### B*64*8*8 --> B*128*4*4 
        x3 = x3 + self.e3_c3(self.e3_c2(x3))   ### B*128*4*4 

        return x1, x2, x3

class Decoder_sep(nn.Module):
    def __init__(self, num_cls):
        super(Decoder_sep, self).__init__()

        self.d2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv2d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):

        de_x3 = self.d2_c1(self.d2(x3))  ###128*4*4-->128*8*8-->64*8*8

        cat_x2 = torch.cat((de_x3, x2), dim=1) ### 128*8*8
        de_x2 = self.d2_out(self.d2_c2(cat_x2))  ###128*8*8-->64*8*8-->64*8*8
        de_x2 = self.d1_c1(self.d1(de_x2))  ###64*8*8 --> 64*16*16 -->32*16*16

        cat_x1 = torch.cat((de_x2, x1), dim=1)  ###64*16*16
        de_x1 = self.d1_out(self.d1_c2(cat_x1))   ###64*16*16 --> 32*16*16 -->32*16*16

        logits = self.seg_layer(de_x1)   ###32*16*16 --> class*16*16
        pred = self.softmax(logits)   ###class*16*16

        return pred, de_x1   

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls):
        super(Decoder_fuse, self).__init__()

        num_cls = num_cls
        self.d2_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.seg_d2 = nn.Conv2d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv2d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv2d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.RFM3 = fusion_prenorm(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*2, num_cls=num_cls)


    def forward(self, x1, x2, x3):

        de_x3 = self.RFM3(x3)     ### N*256*4*4-->N*128*4*4
        pred2 = self.softmax(self.seg_d2(de_x3))   #### N*class*4*4
        de_x3 = self.d2_c1(self.up2(de_x3))    #### N*128*4*4--> N*128*8*8--> N*64*8*8

        de_x2 = self.RFM2(x2)  ## N*128*8*8 --> N*64*8*8
        de_x2 = torch.cat((de_x2, de_x3), dim=1)  ## N*128*8*8
        de_x2 = self.d2_out(self.d2_c2(de_x2)) ## N*64*8*8
        pred1 = self.softmax(self.seg_d1(de_x2))  ## N*class*8*8
        de_x2 = self.d1_c1(self.up2(de_x2))   ## N*64*8*8--> N*64*16*16--> N*32*16*16

        de_x1 = self.RFM1(x1)  ## N*64*16*16--> N*32*16*16
        de_x1 = torch.cat((de_x1, de_x2), dim=1)  ## N*64*16*16
        de_x1 = self.d1_out(self.d1_c2(de_x1)) ## N*64*16*16-->N*32*16*16-->N*32*16*16

        logits = self.seg_layer(de_x1)  ## N*32*16*16 --> N*class*16*16
        pred = self.softmax(logits)  ## N*class*16*16

        return pred, (self.up2(pred1), self.up4(pred2))


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)   ###B*heads*128*128
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   ###B*128*16
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)   ####B*128*16
        return x



class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        num_cls= opt['num_class']
        self.modal1_conv = nn.Conv2d(in_channels=3, out_channels=1,kernel_size=1, stride=1, padding=0)
        self.model1_encoder = Encoder()
        self.modal2_up = nn.Upsample(scale_factor=3, mode='linear', align_corners=True)
        self.modal2_cov = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,2))
        self.model2_encoder = Encoder()

        ########### IntraFormer
        self.modal1_Linear = nn.Conv1d(basic_dims*8, basic_dims*8, kernel_size=1, stride=1, padding=0)
        self.modal2_Linear = nn.Conv1d(basic_dims*8, basic_dims*8, kernel_size=1, stride=1, padding=0)

        self.modal1_pos = nn.Parameter(torch.zeros(1, transformer_basic_dims, basic_dims))   ####1*128*16
        self.modal2_pos = nn.Parameter(torch.zeros(1, transformer_basic_dims, basic_dims))

        self.modal1_transformer = Transformer(embedding_dim=basic_dims, depth=depth, heads=num_heads, mlp_dim=basic_dims)
        self.modal2_transformer = Transformer(embedding_dim=basic_dims, depth=depth, heads=num_heads, mlp_dim=basic_dims)
        ########### IntraFormer
        ########### InterFormer
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, modal1, modal2,is_training):
        #extract feature from different layers
        modal1 = modal1.float()
        modal2 = modal2.float()
        modal1_conv = self.modal1_conv(modal1.reshape(-1, modal1.shape[2],modal1.shape[3],modal1.shape[4])) #### N*16*3*32*32--> (N*16)*3**32*32
        modal1_x1, modal1_x2, modal1_x3 = self.model1_encoder(modal1_conv.reshape(modal1.shape[0], modal1.shape[1], modal1.shape[3],modal1.shape[4]))  ### N*16*32*32

        up_modal2 = self.modal2_up(modal2)   ### N*32*10 -->N*32*30
        modal2_upconv = self.modal2_cov (up_modal2.reshape(up_modal2.shape[0], 1, -1, up_modal2.shape[2]))  ### N*1*32*30-->N*16*32*32
        modal2_x1, modal2_x2, modal2_x3 = self.model2_encoder(modal2_upconv)

        if is_training:
            # modal1_pred, de_modal1 = self.decoder_sep(modal1_x1, modal1_x2, modal1_x3)
            # modal1_L2 = torch.norm(modal1_x1 - de_modal1, p =2) 
            modal2_pred, de_modal2 = self.decoder_sep(modal2_x1, modal2_x2, modal2_x3)
            modal2_L2 = torch.norm(modal2_x1 - de_modal2, p =2) 
        ########### InterFormer
        
        if is_training:
            return modal2_pred, modal2_L2   ###modal2_pred, modal2_L2
        return modal2_x3
