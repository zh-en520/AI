"""
demo04_reco.py 推荐引擎
"""
import json
import numpy as np

with open(
	'../ml_data/ratings.json', 'r') as f:
	ratings = json.loads(f.read())

# scmat存储相似度得分矩阵
users, scmat = list(ratings.keys()), []

for user1 in users:
	scrow = []
	for user2 in users:
		movies = set()
		# 把两个人都看过的电影存入movies中
		for movie in ratings[user1]:
			if movie in ratings[user2]:
				movies.add(movie)
		# 二人没有共同语言
		if len(movies) == 0:
			score = 0
		else: 
			# 分别存储两人对同一部电影的打分
			x1, x2 = [], []
			for movie in movies:
				x1.append(ratings[user1][movie])
				x2.append(ratings[user2][movie])
			x1 = np.array(x1)
			x2 = np.array(x2)
			# score = 1/(1+np.sqrt(((x1-x2)**2).sum()))
			score = np.corrcoef(x1, x2)[0,1]
		scrow.append(score)
	scmat.append(scrow)
print(np.round(scmat, 2))

# 按照相似度从高到低排列每个用户的相似用户
scmat = np.array(scmat)
users = np.array(users)
for i, user in enumerate(users):
	# 按照相似度排序 获取降序的索引
	sorted_indices = scmat[i].argsort()[::-1]
	print('取出前：',sorted_indices)
	sorted_indices = \
		sorted_indices[sorted_indices!=i]
	print('取出后：',sorted_indices)
	# 获取排序后的相似用户
	similar_users = users[sorted_indices]
	# 获取排序后的相似用户的得分
	similar_scores = scmat[i, sorted_indices]
	print('当前相似分数：',similar_scores)

	# 找到所有皮尔逊相关系数正相关的用户
	positive_mask = similar_scores > 0 
	similar_users = \
		similar_users[positive_mask]
	# 遍历所有相似用户 整理推荐电影字典
	# dict={'name':[5.0, 3.5, 2.5], ...}
	recomm_movies = {}
	for i, similar_user in enumerate(
			similar_users):
		# 获取相似用户看过但当前用户没看过
		for movie, score in \
			ratings[similar_user].items():
			if movie not in ratings[user].keys():
				if movie not in recomm_movies:
					recomm_movies[movie]=[score]
				else:
					recomm_movies[movie].append(score)

	#对recomm_movies 进行排序
	movie_list=sorted(recomm_movies.items(), 
		key=lambda x: np.average(x[1]), 
		reverse=True)
	print(user)
	print(movie_list)
	
