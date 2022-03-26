#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
https://github.com/AnTao97/dgcnn.pytorch/tree/97785863ff7a82da8e2abe0945cf8d12c9cc6c18
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_cluster import fps
from torch_cluster import knn as torch_cluster_knn
from helper_tool1 import ConfigPointCloud as cfg

class bilateral_context_block(nn.Module):
    def __init__(self, in_c=32, in_c2=3, cfg=None):
        super(bilateral_context_block, self).__init__()

        self.config = cfg
        self.d_out = torch.Tensor(cfg.d_out)
        self.k_n = cfg.k_n


        #self.in_c = 32
        #self.in_c2 = 3
        self.d_out0 = int(self.d_out[0].item())
        self.in_c = in_c
        self.in_c2 = in_c2
        #self.d_out0 = self.d_cout
        print('self.in_c: ', self.in_c)
        print('self.in_c2: ', self.in_c2)
        print('self.d_out0: ', self.d_out0)

        self.conv1 = nn.Conv2d(self.in_c, self.d_out0//2, 1)
        self.conv2 = nn.Conv2d(self.d_out0, self.in_c2, 1)
        self.conv3 = nn.Conv2d(self.in_c2*3, self.d_out0//2, 1)
        self.conv4 = nn.Conv2d(self.in_c2*3, self.d_out0//2, 1)
        self.conv5 = nn.Conv2d(int(3*self.d_out0/2), self.d_out0//2, 1)
        self.conv6 = nn.Conv2d(self.d_out0, self.d_out0, 1)
        self.conv7 = nn.Conv2d(2*self.d_out0, self.d_out0, 1)
        self.conv8 = nn.Conv2d(self.d_out0, 2*self.d_out0, 1)

        self.bn1 = nn.BatchNorm2d(self.d_out0//2)
        self.bn2 = nn.BatchNorm2d(self.in_c2)
        self.bn3 = nn.BatchNorm2d(self.d_out0//2)
        self.bn4 = nn.BatchNorm2d(self.d_out0//2)
        self.bn5 = nn.BatchNorm2d(self.d_out0//2)

        self.bn7 = nn.BatchNorm2d(self.d_out0)
        self.bn8 = nn.BatchNorm2d(2*self.d_out0)


    def forward(self, feature, input_xyz, input_neigh_idx):
        # bilateral context encoding starts here
        ## f_encoder_i, new_xyz = self.bilateral_context_block(feature, input_xyz, input_neigh_idx, d_out[i],
        #batch_size = input_xyz.size()[0]
        num_points = input_xyz.size()[1]

        #print('batch size: ', batch_size)
        print('num_points: ', num_points)
        

        print('1 feature size: ', feature.size())
        # input encoding
        # make this conversion for tensorflow to torch tensor conversion
        #feature = feature.permute(2,1,0).unsqueeze(0)
        feature = F.relu(self.bn1(self.conv1(feature))) # B * N * 1 * d_out/2 | torch version: B, d_out/2, 1, N
        print('2 feature size: ', feature.size())
        #feature = feature.permute(3,1,0,2) # N, d_out/2, 1, 1 in torch
        feature = feature.permute(0,3,2,1) # N, d_out/2, 1, 1 in torch
        print('3 feature size: ', feature.size())
        #print('input_xyz size: ', input_xyz.size())
        #print('input_neigh_idx size: ', input_neigh_idx.size())


        # bilateral augmentation
        print('input_xyz size: ', input_xyz.size())
        print('input_neigh_idx size: ', input_neigh_idx.size())
        input_neigh_idx = input_neigh_idx.unsqueeze(2)
        print('input_neigh_idx size: ', input_neigh_idx.size())
        a, b, _ = input_neigh_idx.size()
        _, _, c = input_xyz.size()
        input_neigh_idx1 = input_neigh_idx.expand((a, b, c))
        neigh_xyz = torch.gather(input_xyz, 1, input_neigh_idx1) 
        print('neigh_xyz size: ', neigh_xyz.size())
        #print('neigh_xyz[0]: ', neigh_xyz[0])
        _, _, c, d = feature.size()
        input_neigh_idx2 = input_neigh_idx.expand((a, b, c))
        input_neigh_idx2 = input_neigh_idx2.unsqueeze(3)
        input_neigh_idx2 = input_neigh_idx2.expand((a, b, c, d))
        neigh_feat = torch.gather(feature, 1, input_neigh_idx2) 
        print('4 neigh_feat size: ', neigh_feat.size())
        ################33
        #neigh_feat = neigh_feat.reshape(-1, neigh_feat.size()[0], neigh_feat.size()[2])
        #neigh_feat = neigh_feat.reshape(1, -1, self.k_n, neigh_feat.size()[2])
        #print('neigh_feat size: ', neigh_feat.size())
        
        #raise ValueError("Exit!")
        # bilateral augmentation

        tile_feat = feature.tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        #tile_xyz = input_xyz.tile(1, 1, self.k_n, 1) # B * N * k * 3
        print('xyz size: ', input_xyz.size())
        tile_xyz = torch.unsqueeze(input_xyz, 2).tile(1, 1, self.k_n, 1) # B * N * k * 3

        print('tile_xyz size: ', tile_xyz.size())
        print('tile_feat size: ', tile_feat.size())


        # we are making every dim in pytorch convention
        tile_xyz = tile_xyz.permute(0, 3, 2, 1) # B, 3, k, N
        tile_feat = tile_feat.permute(0, 3, 2, 1) # B, d_out/2, k, N
        print('after permute tile_xyz size: ', tile_xyz.size())
        print('after permute tile_feat size: ', tile_feat.size())

        a, b, c = neigh_xyz.size() 
        neigh_xyz = neigh_xyz.reshape(a, c, -1,  num_points) # B, 3, k, N
        a, b, c, d = neigh_feat.size() 
        neigh_feat = neigh_feat.reshape(a, d, -1, num_points)  # B, d_out/2, k, N
        print('after permute neigh_xyz size: ', neigh_xyz.size())
        print('after permute neigh_feat size: ', neigh_feat.size())

        feat_info = torch.cat((neigh_feat - tile_feat, tile_feat), dim=1) # B, d_out, k, N
        print('feat_info size : ', feat_info.size())
        neigh_xyz_offsets = F.relu(self.bn2(self.conv2(feat_info))) # B, 3, k, N
        print('neigh_xyz_offsets size: ', neigh_xyz_offsets.size())
        #raise ValueError("Exit!")
        shifted_neigh_xyz = neigh_xyz + neigh_xyz_offsets # B * N * k * 3
        print('shifted_neigh_xyz size: ', shifted_neigh_xyz.size())


        xyz_info = torch.cat((neigh_xyz - tile_xyz, shifted_neigh_xyz, tile_xyz), dim=1) # B, 9, k, N
        print('xyz_info size : ', xyz_info.size())
        neigh_feat_offsets = F.relu(self.bn3(self.conv3(xyz_info))) # B, d_out/2,  k, N 
        print('neigh_feat_offsets size: ', neigh_feat_offsets.size())
        shifted_neigh_feat = neigh_feat + neigh_feat_offsets # B, D_out/2, k, N
        print('shifted_neigh_feat size: ', shifted_neigh_feat.size())

        xyz_encoding = F.relu(self.bn4(self.conv4(xyz_info))) # B, d_out/2, k, N
        print('xyz_encoding size : ', xyz_encoding.size())
        feat_info = torch.cat((shifted_neigh_feat, feat_info), dim=1) # B, 3/2 * d_out, k, N
        print('feat_info size : ', feat_info.size())
        feat_encoding = F.relu(self.bn5(self.conv5(feat_info))) # B, d_out/2, k, N
        print('feat_encoding size : ', feat_encoding.size())
        #raise ValueError("Exit!")

        # Mixed Local Aggregation
        overall_info = torch.cat((xyz_encoding, feat_encoding), dim=1) # B, d_out, k, N
        print('overall_info size : ', overall_info.size())
        k_weights = self.conv6(overall_info) # B, d_out, k, N
        print('k_weights size : ', k_weights.size())
        k_weights= torch.nn.Softmax(dim=2)(k_weights) # B, d_out, k, N
        print('k_weights size : ', k_weights.size())
        overall_info_weighted_sum = torch.sum(overall_info * k_weights, dim=2, keepdim=True) # B, d_out, 1, N
        print('overall_info_weighted_sum size : ', overall_info_weighted_sum.size())
        overall_info_max = torch.max(overall_info, dim=2, keepdim=True)[0] # B, d_out, 1, N
        #overall_info_max = torch.max(overall_info, dim=2, keepdim=True) # B, d_out, 1, N
        print('overall_info_max size : ', overall_info_max.size())
        #print('overall_info_max : ', overall_info_max)
        overall_encoding = torch.cat((overall_info_max, overall_info_weighted_sum), dim=1) # B, 2*d_out, 1, N
        print('overall_encoding size : ', overall_encoding.size())

        overall_encoding = F.relu(self.bn7(self.conv7(overall_encoding))) # B, d_out, 1, N
        print('overall_encoding size : ', overall_encoding.size())
        #raise ValueError("Exit!")
        output_feat = F.leaky_relu(self.bn8(self.conv8(overall_encoding))) # B, 2*d_out, 1, N
        print('output_feat size : ', output_feat.size())

        return output_feat, shifted_neigh_xyz

 # we will design a two layer module for the begininning
class MMNet_seg(nn.Module):
    def __init__(self, num_classes=15, num_channels=15, cfg=None):
        super(MMNet_seg, self).__init__()


        self.config = cfg
        self.d_out = torch.Tensor(cfg.d_out)
        self.ratio = torch.Tensor(cfg.sub_sampling_ratio).cuda()
        self.k_n = cfg.k_n
        self.num_layers = cfg.num_layers
        self.n_pts = cfg.num_points

        # we have the output as 32 for the linear layer
        # because we have originally 24 dims and we want to
        # take them ti higher dims, whereas originally
        # in the authors code they might have 6 dims
        # which are being transformed to 8 dim
        self.fc0 = nn.Linear(24, 32)
        self.bn0 = nn.BatchNorm1d(32)

        self.in_c = 32
        self.in_c2 = 3
        self.d_out0 = int(self.d_out[0].item())
        print('self.in_c: ', self.in_c)
        print('self.in_c2: ', self.in_c2)
        print('self.d_out0: ', self.d_out0)


        self.bcb1 = bilateral_context_block(self.in_c, self.in_c2, self.config)

        self.conv1 = nn.Conv2d(2*self.d_out0, 2*self.d_out0, 1)
        self.up_conv1 = nn.ConvTranspose2d(4*self.d_out0, 2*self.d_out0, 1)
        self.conv2 = nn.Conv2d(2*self.d_out0, 1, 1)
        #self.conv3 = nn.Conv2d(2*self.d_out0, 4*self.d_out0, 1)
        self.conv3 = nn.Conv2d(2*self.d_out0, 64, 1)
        self.conv4 = nn.Conv2d(64, 32, 1)
        self.conv5 = nn.Conv2d(32, self.config.num_classes, 1)

        self.bn1 = nn.BatchNorm2d(2*self.d_out0)
        self.up_bn1 = nn.BatchNorm2d(2*self.d_out0)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        #self.bn3 = nn.BatchNorm2d(4*self.d_out0)
        #self.bn4 = nn.BatchNorm2d(2*self.d_out0)

    def forward(self, x):
        #self.feature = inputs['features']
        # X is B x D x N
        # interestingly in baafnet code, the feature is using all the features
        # including the coordinates as well , strange!
        #feature = x.clone()
        feature = x
        og_xyz = x[:, 9:12, :] # B*3*N we treat the barycenters as the pointcloud
        batch_sz, dims, num_pts = og_xyz.size()
        # everything else is considered feature includeing the vertex coordinates for now
        #og_xyz = feature[:, :, :3]
        #feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature_orig = feature.clone() # B, D, N
        print('1111 feature size: ', feature.size())
        feature = feature.reshape(-1, feature.shape[1]) # B*N, D
        print('2222 feature size: ', feature.size())
        feature = F.leaky_relu(self.bn0(self.fc0(feature))) # B*N, D -> B*N, D'
        print('3333 feature size: ', feature.size())
        feature = torch.unsqueeze(feature, axis=1) # B*N, 1, D'
        print('4444 feature size: ', feature.size())
        #feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        #feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        print('og_xyz size: ', og_xyz.size()) # B,3,N
        #input_xyz = og_xyz.view(-1, og_xyz.size()[1])
        #input_xyz = og_xyz.reshape(-1, og_xyz.size()[1])
        
        input_xyz = og_xyz.reshape(batch_sz, num_pts, dims)
        print('input_xyz size: ', input_xyz.size()) # N*3
        # we will deal with the batch size using loops because anyway the batch size is not going
        # to be huge soon, TBD: we should replace this function with the batch operation
        # sometime later

        # re arrange the neighbor indices in tensor format
        input_neigh_idx = self.batch_knn(input_xyz, input_xyz, self.k_n)
        print('input_neigh_idx size: ', input_neigh_idx.size())


        print('input_xyz size: ', input_xyz.size()) # N*3
        input_up_samples = []
        new_xyz_list = []
        xyz_list = []
        print('5555 feature size: ', feature.size())
        feature = feature.permute(0, 2, 1).view(batch_sz, -1, 1, num_pts) 
        print('6666 feature size: ', feature.size())
        

        i = 0
        n_pts = self.n_pts // self.ratio[i] # N / r
        ratio = (1./self.ratio[i]).cuda() # N / r

        # calculate batchwise fps instead of individual fps
        inputs_sub_idx = self.batch_fps(input_xyz, ratio)
        print('inputs_sub_idx size: ', inputs_sub_idx.size())
        

        # perform gather operation
        #inputs_sub_idx = inputs_sub_idx.unsqueeze(-1)
        a, b = inputs_sub_idx.size()
        #a, b, _ = inputs_sub_idx_idx.size()
        _, _, c = input_xyz.size()
        #pool_idx = pool_idx.expand((1, 50, 32))
        #interp_idx = interp_idx.expand((a, b, c))

        sub_xyz = torch.gather(input_xyz, 1, inputs_sub_idx.unsqueeze(-1).expand((a, b, c)))
        #sub_xyz = input_xyz[inputs_sub_idx] # (N / r) * 3 (3 for the three coordinates)
        print('sub_xyz size: ', sub_xyz.size())


        # get the sub_xyz from the original input_xyz with the help of the inputs_sub_idx
        #inputs_interp_idx = torch_cluster_knn(input_xyz, sub_xyz, 1) # 2 * (N/r)
        # for every point in the original input_xya find the nearest neighbor
        # in the  sub_xyz, this will be used while upsampling
        inputs_interp_idx = self.batch_knn(sub_xyz, input_xyz, 1) # 2 * (N/r)
        #print('inputs_interp_idx: ', inputs_interp_idx)
        print('inputs_interp_idx size: ', inputs_interp_idx.size())
        input_up_samples.append(inputs_interp_idx)


        # bilateral context encoding starts here
        f_encoder_i, new_xyz = self.bcb1(feature, input_xyz, input_neigh_idx)
        print('f_encoder_i size: ', f_encoder_i.size())
        print('inputs_sub_idx size: ', inputs_sub_idx.size())

        # get the feature vector from the indices which we initially subsampled
        # so that feature correspond to the points are chosen
        # but why do we take the max of those
        f_sampled_i = self.random_sample(f_encoder_i, inputs_sub_idx) # B, d_out*2, (N/r), 1
        print('f_sampled_i size: ', f_sampled_i.size())
        if i == 0:
            f_encoder_list.append(f_encoder_i)
        f_encoder_list.append(f_sampled_i)
        xyz_list.append(input_xyz)
        input_xyz = sub_xyz

        ######################### Encoder ########################

        ######################### Decoder ########################
        # Adaptive fusion module

        f_multi_decoder = [] # full sized feature maps
        f_weights_decoders = [] # point-wise adaptive fusion weights

        n = 0
        feature = f_encoder_list[-1 - n]
        feature = F.leaky_relu(self.bn1(self.conv1(feature)))
        print('feature size: ', feature.size())

        f_decoder_list = []
        j = 0
        f_interp_i = self.nearest_interpolation(feature, input_up_samples[- j - 1 - n])
        ip = torch.cat([f_encoder_list[-j - 2 -n], f_interp_i], dim=1)
        print('ip size: ', ip.size())
        f_decoder_i = F.leaky_relu(self.up_bn1(self.up_conv1(ip)))
        feature = f_decoder_i
        f_decoder_list.append(f_decoder_i)
        print('feature size: ', feature.size())
        # collect full-sized feature maps which are upsampled from multiple resolu
        f_multi_decoder.append(f_decoder_list[-1])
        # summarize the point level information
        curr_weight = self.conv2(f_decoder_list[-1])
        print('curr weight size: ', curr_weight.size())
        f_weights_decoders.append(curr_weight)

        # regress the fusion parameters
        f_weights = torch.cat(f_weights_decoders, dim=1) # the concatenation should be along channel
        print('f_weights size: ', f_weights.size())
        f_weights = F.softmax(f_weights, dim=1)
        print('f_weights size: ', f_weights.size())

        # adaptively fuse them by calculating a weighted sum
        f_decoder_final = torch.zeros_like(f_multi_decoder[-1])
        for i in range(len(f_multi_decoder)):
            #f_decoder_final = f_decoder_final + torch.tile(tf.expand_dims(f_weights[:,i,:,:], dim=1), [1, 1, 1, f_multi_decoder[i].get_shape()[-1].value]) * f_multi_decoder[i]
            f_decoder_final = f_decoder_final + torch.tile(torch.unsqueeze(f_weights[:,i,:,:], dim=1), [1, f_multi_decoder[i].size()[1], 1, 1]) * f_multi_decoder[i]
            print('i: {} f_decoder_final size: {}'.format(i, f_decoder_final.size()))


        f_layer_fc1 = F.leaky_relu(self.bn3(self.conv3(f_decoder_final)))    
        print('f_layer_fc1 size: ', f_layer_fc1.size())
        f_layer_fc2 = F.leaky_relu(self.bn4(self.conv4(f_layer_fc1)))    
        print('f_layer_fc2 size: ', f_layer_fc2.size())
        f_layer_drop = F.dropout(f_layer_fc2, p=0.5)
        print('f_layer_drop size: ', f_layer_drop.size())
        f_layer_fc3 = self.conv5(f_layer_drop)    
        print('f_layer_fc3 size: ', f_layer_fc3.size())

        f_out = torch.squeeze(f_layer_fc3, dim=2)
        print('f_out size: ', f_out.size())

        x = F.log_softmax(f_out, dim=1)
        x = x.permute(0, 2, 1)

        return x
        #raise ValueError("Exit!")
        #return f_out, new_xyz_list, xyz_list


    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        
        #feature = feature.squeeze(2)
        print('interp_idx size ', interp_idx.size())
        feature = feature.squeeze(3)
        print('nearest interpolation: feature size: ', feature.size())
        batch_size = feature.size()[0]
        #batch_size = interp_idx.size()[0]
        print('nearest interpolation: batch size: ', batch_size)
        #up_num_points = interp_idx.size()[1]
        up_num_points = interp_idx.size()[1]
        print('nearest interpolation: up_num_points: ', up_num_points)
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        print('nearest interpolation: interp_idx size: ', interp_idx.size())

        feature = feature.permute(0, 2, 1)
        #pool_idx = pool_idx.unsqueeze(-1)
        #a, b, _ = pool_idx.size()
        interp_idx = interp_idx.unsqueeze(-1)
        a, b, _ = interp_idx.size()
        _, _, c = feature.size()
        #pool_idx = pool_idx.expand((1, 50, 32))
        interp_idx = interp_idx.expand((a, b, c))

        interpolated_features = torch.gather(feature, 1, interp_idx)
        print('nearest interpolation: interpolated feature size: ', interpolated_features.size())
        interpolated_features = interpolated_features.unsqueeze(dim=2)
        print('nearest interpolation: interpolated feature size: ', interpolated_features.size())
        #interpolated_features = interpolated_features.permute(0, 3, 1, 2)
        interpolated_features = interpolated_features.permute(0, 3, 2, 1)
        print('nearest interpolation: interpolated feature size: ', interpolated_features.size())
        return interpolated_features


    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = torch.squeeze(feature, dim=2)
        #pool_idx = inputs_sub_idx
        num_neigh = pool_idx.size()[1]
        batch_size = feature.size()[0]
        pool_idx = pool_idx.reshape(batch_size, -1)
        print('feature size: ', feature.size())
        print('num_neigh : ', num_neigh)
        print('batch_size: ', batch_size)
        print('pool_idx size: ', pool_idx.size())

        feature = feature.permute(0, 2, 1)
        pool_idx = pool_idx.unsqueeze(-1)
        a, b, _ = pool_idx.size()
        _, _, c = feature.size()
        #pool_idx = pool_idx.expand((1, 50, 32))
        pool_idx = pool_idx.expand((a, b, c))
        #pool_idx = pool_idx.unsqueeze(-1).expand_as(f_encoder_i)
        print('feature size: ', feature.size())
        print('pool_idx size: ', pool_idx.size())
        pool_features = torch.gather(feature, 1, pool_idx) 
        #pool_features = 
        print('pool features size: ', pool_features.size())
        pool_features = pool_features.reshape(batch_size, c, num_neigh, -1)
        print('after reshape pool features size: ', pool_features.size())
        #pool_features = torch.max(pool_features, dim=2, keepdims=True)
        #print('after torch max pool features size: ', pool_features.size())
        #print('after torch max pool features: ', pool_features)
        # I do not know why should we take max here, commenting it out for now
        # is the purpose to reduce the number of channels?
        # the only max tht makes sense to me is along the feature dimension
        '''
        pool_features = torch.max(pool_features, dim=2, keepdims=True)[0]
        print('after torch max pool features size: ', pool_features.size())
        '''
        return pool_features
        
    @staticmethod
    def batch_fps(x, r):
        batch_sz, _, __ = x.size()
        all_tensors = []
        for i in range(batch_sz):
            cur_tensor = x[i]
            print('cur_tensor size: ', cur_tensor.size())
            cur_sub_idx = fps(cur_tensor, ratio = r) # N / r
            print('cur_sub_idx size: ', cur_sub_idx.size())
            # append all neighbors for a particular sample
            all_tensors.append(cur_sub_idx)

        # re arrange the neighbor indices in tensor format
        inputs_sub_idx = torch.stack(all_tensors, dim=0)
        #print('inputs_sub_idx size: ', inputs_sub_idx.size())
        return inputs_sub_idx

    @staticmethod
    def batch_knn(x, y, k):
        batch_sz, _, _ = x.size()
        all_tensors = []
        for i in range(batch_sz):
            cur_tensor = x[i]
            cur_tensor2 = y[i]
            print('cur_tensor size: ', cur_tensor.size())
            cur_neighbor = torch_cluster_knn(cur_tensor, cur_tensor2, k)[1]
            print('cur_neighbor size: ', cur_neighbor.size())
            # append all neighbors for a particular sample
            all_tensors.append(cur_neighbor)

        # re arrange the neighbor indices in tensor format
        input_neigh_idx = torch.stack(all_tensors, dim=0)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        return input_neigh_idx


if __name__ == "__main__":
    # A full forward pass
    inputs = torch.randn(8, 24, 100)
    inputs = inputs.cuda()
    model = MMNet_seg(num_classes=8, num_channels=24, cfg=cfg)
    model = model.cuda()
    outputs = model(inputs)
    # print(x.shape)
    del model
    del outputs
    # print(x.shape)

