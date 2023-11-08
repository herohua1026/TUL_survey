#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import operator
import sys
import gc
import re
# listdir
from os import listdir
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.coo import coo_matrix

from fileinput import filename
import math
#import matplotlib
#import matplotlib.pyplot as plt
import os

from config import *


table_X = {}
table_gps = {}

def get_index(userT):
    userT = list(set(userT))
    User_List = userT
    return User_List

def read_train_data():
    """
    只读入原始训练数据，数据并未经过embedding
    :return: 分割好的train和test
    """
    test_T = list()
    test_UserT = list()
    test_lens = list()
    ftraindata = open(data_input_path+data_input, 'r')  # gowalla_scopus_1006.dat
    tempT = list()  #
    pointT = list()  #
    userT = list()  # User ID
    seqlens = list()  #
    item = 0
    # for line in ftraindata.readlines():
    #     lineArr = line.split()
    #     X = list()
    #     for i in lineArr:
    #         X.append(int(i))
    #     tempT.append(X)
    #     userT.append(X[0])
    #     pointT.append(X[1:])
    #     seqlens.append(len(X) - 1)  #
    #     item += 1
#################################################
    test = list()
    pointtt = list()
    item = 0
    count = 1
    line_num = 0
    for line in ftraindata.readlines():
        #line = line.replace('\r\n', '')
        #lineArr = line.split(' ')
        lineArr = line.strip().split(' ')
        userT.append(lineArr[0])
        for i in range(1, len(lineArr)):
            
            if count == 1:
                test.append(lineArr[i])
                pointtt.append(lineArr[i])
                count = count + 1
            elif count == 3:
                test.append((lineArr[i]))
                tempT.append(test)
                count = 1
                test = []
            elif count == 2:
                test.append(lineArr[i])
                count = count + 1
            else:
                #print(lineArr[0],i,'___'+lineArr[i]+"___",len(lineArr[i]),count,line_num)
                test.append(float(lineArr[i]))
                count = count + 1
            line_num += 1
        pointT.append(tempT)
        pointtt = []
        seqlens.append((len(lineArr) - 1) / 4)
        item = item + 1
        test = []
        tempT = []

##################################################

    # Test 98481
    Train_Size = 7800
    pointT = pointT[:Train_Size]
    userT = userT[:Train_Size]
    seqlens = seqlens[:Train_Size]
    User_List = get_index(userT)
    print('Count average length of trajectory')
    avg_len = 0
    for i in range(len(pointT)):
        avg_len += len(pointT[i])

    print('--------Average Length=', avg_len / (len(pointT)))
    print(User_List)
    print(len(User_List))
    print("Index numbers", len(User_List))
    print("point T", pointT[Train_Size - 1])
    flag = 0
    count = 0
    temp_pointT = list()
    temp_userY = list()
    temp_seqlens = list()
    User = 0  #
    rate = 0.7  # 10% for test

    # 对于一个用户的轨迹，找到最后的10%作为test，剩下的作为train，用户id为70的只有一条轨迹，无法分割test，因此只有train
    for index in range(len(pointT)):
        if (userT[index] != flag or index == (len(pointT) - 1)):  # 当前用户不是flag 或者 找到pointT的最后一个，进入
            User += 1
            # split data
            if (count > 1):  # 之前一直连续匹配到flag的数目，如果这个用户只有一条轨迹的话，只有train，没有test
                # print "count",count," ",index
                test_T += (pointT[int((index - math.ceil(count * rate))):index])  # test point
                test_UserT += (userT[int((index - math.ceil(count * rate))):index])  # test user
                test_lens += (seqlens[int((index - math.ceil(count * rate))):index])  # test seq len
                temp_pointT += (pointT[int((index - count)):int((index - count * rate))])
                temp_userY += (userT[int((index - count)):int((index - count * rate))])
                temp_seqlens += (seqlens[int((index - count)):int((index - count * rate))])
            else:
                temp_pointT += (pointT[int((index - count)):int((index))])
                temp_userY += (userT[int((index - count)):int((index))])
                temp_seqlens += (seqlens[int((index - count)):int((index))])
            count = 1  #
            flag = userT[index]  #
        else:
            count += 1

    pointT = temp_pointT
    userT = temp_userY
    seqlens = temp_seqlens
    print('training Numbers=', item - 1)
    print('div number=', Train_Size)
    print('Train Size=', len(userT), ' Test Size=', len(test_UserT), "User numbers=", len(User_List))
    print(test_T[-1])

    return tempT, pointT, userT, seqlens, test_T, test_UserT, test_lens, User_List  #

##### word2vector embedding ####
# def get_xs():  #
#     """
#     相当于读取point的embedding，存到table_X中进行查找
#     :return: 读取到的embedding向量
#     """
#     fpointvec = open("C:\\Users\\uqhshi1\\Desktop\\TULER-VARIANTS\\Foursquare_6h\\NYC_6h_cbow.dat", 'r')  #gowalla_vector_250d.dat  gowalla_em_250.dat
#     #     table_X={}  #
#     item = 0
#     for line in fpointvec.readlines():
#         lineArr = line.split()
#         if len(lineArr) < 250 or lineArr[0] == '</s>':
#             continue
#         item += 1  #
#         X = list()
#         for i in lineArr[1:]:
#             X.append(float(i))  #
#         table_X[int(lineArr[0])] = X
#     print("point number item=", item)
#     print(table_X[-2605535307058307208])
#
#     return table_X


##### GNN-SAGE  ####
def lonlat2meters(lon, lat):
    x = lon * 20037508.34 / 180
    try:
        y = math.log(math.tan((90 - 1e-10 + lat) * math.pi / 360)) / (math.pi / 180)
    except:
        y = math.log(math.tan((90 + 1e-10 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y

class GraphCenter():
    def __init__(self, train_trajs, test_trajs, userT, test_UserT, spatio=500):

        self.spatio = spatio
        self.train_trajs = train_trajs
        self.test_trajs = test_trajs
        self.userT = userT
        self.test_UserT = test_UserT

        self.vocabs = None

        self.adj_lists = None
        self.target_num = None
        self.feats = None

    def build_vocabs(self, trajs, vocabs):
        for traj in trajs:
            for i in range(len(traj)):
                point = traj[i][0]
                if point not in vocabs:
                    vocabs[point] = len(vocabs)
        # return vocabs

    def convert_trajs_to_vocab(self, trajs):
        converted_trajs = []
        for traj in trajs:
            converted_traj = []
            # for point in traj:
            for i in range(len(traj)):
                point = traj[i][0]
                if point not in self.vocabs:
                    converted_traj.append(self.vocabs["UNK"])
                else:
                    converted_traj.append(self.vocabs[point])
            converted_trajs.append(converted_traj)
        return converted_trajs

    def read_vidx_to_latlon(self):
        vidx_to_latlon = dict()

        for traj in self.train_trajs + self.test_trajs:
            traj_gps = []
            for i in range(len(traj)):
                # vidx, lat, lon = traj[i][0], traj[i][2], traj[i][3]
                # vidx_to_latlon[int(vidx)] = [float(lat), float(lon)]

                vidx, lat, lon = traj[i][0], traj[i][1], traj[i][2]
                vidx_to_latlon[str(vidx)] = [float(lat), float(lon)]

        return vidx_to_latlon

    def construct_spatial_graph(self):
        revocabs = dict([[idx, vocab] for vocab, idx in self.vocabs.items()])
        POI_max = max(revocabs.keys())
        POIlatlon = []
        for i in range(2):
            POIlatlon.append([0, 0])
        for i in range(2, POI_max + 1):
            #item = int(revocabs[i])

            item = str(revocabs[i])

            if item not in self.vidx_to_latlon.keys():
                POIlatlon.append([0, 0])
            else:
                POIlatlon.append(lonlat2meters(self.vidx_to_latlon[item][1], self.vidx_to_latlon[item][0]))
        POIneighbor_dist = pairwise_distances(POIlatlon, n_jobs=4)
        POIneighbor_graph = np.where(POIneighbor_dist < self.spatio, 1, 0)
        spaital_graph = coo_matrix(POIneighbor_graph)
        return np.stack([spaital_graph.row, spaital_graph.col], 0)

    def construct_transition_graph(self):
        Vgraph = dict()
        # for traj in self.train_trajs: # No self.val_trajs
        for traj in self.train_poi_trajs + self.test_poi_trajs:
            for i in range(len(traj) - 1):
                p1, p2 = traj[i], traj[i + 1]
                if p1 not in Vgraph:
                    Vgraph[p1] = set()
                if p2 not in Vgraph:
                    Vgraph[p2] = set()
                Vgraph[p1].add(p2)
                Vgraph[p2].add(p1)
        return Vgraph

    def convert_trajs_to_vocab(self, trajs):
        converted_trajs = []
        for traj in trajs:
            converted_traj = []
            # for point in traj:
            for i in range(len(traj)):
                point = traj[i][0]
                if point not in self.vocabs:
                    converted_traj.append(self.vocabs["UNK"])
                else:
                    converted_traj.append(self.vocabs[point])
            converted_trajs.append(converted_traj)
        return converted_trajs

    def fusion_svgraph(self):
        for p1, p2 in self.spatial_graph.transpose():
            if p1 not in self.visiting_graph:
                self.visiting_graph[p1] = set()
            if p2 not in self.visiting_graph:
                self.visiting_graph[p2] = set()
            self.visiting_graph[p1].add(p2)
            self.visiting_graph[p2].add(p1)
        self.adj_lists = dict([(node, list(self.visiting_graph[node])) for node in self.visiting_graph])
        self.adj_lists[0] = [0]
        self.adj_lists[1] = [1]

    def load_graphSet(self):
        # load train_trajs,test_trajs,test_T, userT,test_UserT,

        self.test_uid_list = self.test_UserT
        self.test_trajs = self.test_trajs

        self.train_uid_list = self.userT
        self.train_trajs = self.train_trajs

        #         self.val_uid_list, self.val_trajs = self.read_trajs_file(self.valid_path)
        # all_traj = self.test_trajs + self.train_trajs
        self.vidx_to_latlon = self.read_vidx_to_latlon()
        #         self.test_timestamps_list, self.test_weekdays_list, self.test_hours_list = self.read_trajs_time_file(self.test_time_path)

        # user set for linking
        self.users = dict([(uid, idx) for idx, uid in enumerate(set(self.train_uid_list))])
        self.target_num = len(self.users)

        # vocabs
        self.vocabs = {"PAD": 0, "UNK": 1}
        self.build_vocabs(self.train_trajs, self.vocabs)
        # self.build_vocabs(self.val_trajs, self.vocabs)
        self.build_vocabs(self.test_trajs, self.vocabs)
        print("num of poi = ", len(self.vocabs))
        self.feats = np.eye(len(self.vocabs) + 1)

        self.train_poi_trajs = self.convert_trajs_to_vocab(self.train_trajs)
        # self.val_trajs = self.convert_trajs_to_vocab(self.val_trajs)
        self.test_poi_trajs = self.convert_trajs_to_vocab(self.test_trajs)

        # Load spatial graph
        # if self.graph_path is not None:
        #   _, self.spatial_graph = pickle.load(open(self.graph_path, "rb"))
        # else:
        self.spatial_graph = self.construct_spatial_graph()
        # Constructing visiting graph
        self.visiting_graph = self.construct_transition_graph()
        # fusing the two graph, i.e., spatial graph and visiting graph
        self.fusion_svgraph()


def get_one_hot(i):
    x: list[int] = [0] * n_classes
    x[i] = 1
    return x


def get_mask_index(value, User_List):
    #     print User_List #weikong
    return User_List.index(value)


def get_pvector(i):  #
    return table_X[int(i)]


def get_true_index(index, User_List):
    return User_List[index]





if __name__ == "__main__":
    tempT, pointT, userT, seqlens, test_T, test_UserT, test_lens, User_List = read_train_data()

    print(pointT[0])
    
    print('Pay attention to modify the POI number in the config.py')

    #get_xs()

    #print(table_X[pointT[0][0]])

   