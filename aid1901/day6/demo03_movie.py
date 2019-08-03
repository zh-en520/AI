# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_movie.py  电影推荐
"""
import json
import numpy as np

with open('../ml_data/ratings.json', 'r') as f:
    ratings = json.loads(f.read())

# 整理用户之间的相似度得分矩阵
users, scmat = list(ratings.keys()), []
for user1 in users:
    scrow = []
    for user2 in users:
        # 计算user1与user2的相似度 添加到scrow
        movies = set()
        for movie in ratings[user1]:
            if movie in ratings[user2]:
                movies.add(movie)
        if len(movies) == 0:
            score = 0
        else:
            A, B = [], []
            for movie in movies:
                A.append(ratings[user1][movie])
                B.append(ratings[user2][movie])
            A = np.array(A)
            B = np.array(B)
            # 计算A与B向量的相似度
            # score = 1/(1+np.sqrt(((A-B)**2).sum()))
            score = np.corrcoef(A, B)[0,1]


        scrow.append(score)
        print('scrow:',scrow)
    scmat.append(scrow)

users = np.array(users)
scmat = np.array(scmat)

for scrow in scmat:
    print(' '.join(['{:.2f}'.format(score)\
             for score in scrow]))

# 按照相似度从高到低排列每个用户的相似用户
for i, user in enumerate(users):
    # 获取所有相似用户得分，去掉自己，排序
    sorted_indices = scmat[i].argsort()[::-1]
    sorted_indices = \
        sorted_indices[sorted_indices != i]
    # user的所有相似用户
    sim_users = users[sorted_indices]
    # user所有相似用户的相似度得分
    sim_scores = scmat[i, sorted_indices]
    # print(user, sim_users, sim_scores, sep='\n')

    # 生成推荐清单
    # 正相关得分的掩码
    positive_mask = sim_scores > 0
    # 获取所有正相关用户的用户名
    sim_users = sim_users[positive_mask]
    # 为user构建推荐清单，找到每个sim_user
    # 看过但当前user没看过的电影，存入字典结构
    # 存储推荐清单：
    # {'电影1':[4.0, 5.0], '电影2':[5.0]}
    reco_movies = {}
    for i, sim_user in enumerate(sim_users):
        for movie, score in \
            ratings[sim_user].items():
            # 相似用户看过，但当前用户没看过
            if movie not in ratings[user].keys():
                if movie not in reco_movies:
                    reco_movies[movie] = [score]
                else:
                    reco_movies[movie].append(score)
    print(user)
    # print(reco_movies)
    # 对推荐清单进行排序
    movie_list = sorted(reco_movies.items(), 
        key=lambda x:np.average(x[1]), 
        reverse=True)
    print(movie_list)

    