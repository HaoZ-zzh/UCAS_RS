import numpy as np

def pearson_correlation(x, y, mean_x=None, mean_y=None):
    # 计算x和y的平均值
    if mean_x is None:
        mean_x = np.mean(x)
    if mean_y is None:
        mean_y = np.mean(y)
    
    # 分子：协方差
    covariance = np.sum((x - mean_x) * (y - mean_y))
    
    # 分母：x和y的方差
    variance_x = np.sum((x - mean_x) ** 2)
    variance_y = np.sum((y - mean_y) ** 2)
    
    # 计算标准差
    std_x = np.sqrt(variance_x)
    std_y = np.sqrt(variance_y)
    
    # 计算相关系数
    correlation = 1.0 * covariance / (std_x * std_y)
    
    return correlation

# ratings是用户的评分
ratings = np.array([
    [4, np.nan, 3, 5],
    [np.nan, 5, 4, np.nan],
    [5, 4, 2, np.nan],
    [2, 4, np.nan, 3],
    [3, 4, 5, np.nan]
])

# 每个用户的有效评分的平均数
user_means = np.nanmean(ratings, axis=1)
# print("每个用户的有效评分的平均数:")
# print(user_means)


# 初始化用户之间的相关系数矩阵
user_correlations = np.full((len(ratings), len(ratings)), np.nan)

# 计算用户之间的皮尔逊相关系数
for i in range(len(ratings)):
    for j in range(len(ratings)):
        if i != j:  # 避免对角线元素，即同一用户的自身相关系数
            # 获取两个用户的有效评分下标（排除NaN）
            indices_i = np.where(~np.isnan(ratings[i]))[0]
            indices_j = np.where(~np.isnan(ratings[j]))[0]
            # print(indices_i)
            # print(indices_j)
            
            # 找到两个用户都评分的商品的下标交集
            common_indices = np.intersect1d(indices_i, indices_j)
            # print(common_indices)
            
            # 如果有共同评分的商品，则计算相关系数
            if len(common_indices) > 0:  # 确保有足够的数据点来计算相关系数
                # 获取共同评分的商品的评分值
                ratings_i_common = ratings[i, common_indices]
                ratings_j_common = ratings[j, common_indices]
                # 计算相关系数
                correlation = pearson_correlation(ratings_i_common, ratings_j_common, user_means[i], user_means[j])
                user_correlations[i][j] = correlation
                # print("用户", i, "和用户", j, "之间的相关系数:", correlation)

# 打印用户之间的相关系数矩阵
print("用户之间的相关系数矩阵:")
print(user_correlations)

# 预测每个用户对每个商品的评分
predictions = np.nan * np.ones_like(ratings)  # 初始化预测矩阵为NaN

# 对于每个用户和商品的评分矩阵中的空缺评分进行预测
for user_index in range(len(ratings)):
    for item_index in range(len(ratings[0])):
        if np.isnan(ratings[user_index, item_index]):
            # 对于每个缺失评分，计算所有用户的加权平均评分
            weighted_sum = 0
            weight_sum = 0
            for other_user_index in range(len(ratings)):
                # 计算当前用户和其他用户的相关系数
                correlation = user_correlations[user_index][other_user_index]
                # 如果相关系数不为NaN，并且其他用户对该商品有评分
                if not np.isnan(correlation) and not np.isnan(ratings[other_user_index, item_index]):
                    weighted_sum += correlation * (ratings[other_user_index, item_index] - user_means[other_user_index])
                    weight_sum += abs(correlation)  
            # 如果有有效的邻居，则计算加权平均评分
            if weight_sum > 0:
                predictions[user_index, item_index] = user_means[user_index] + weighted_sum / weight_sum

# 打印预测的评分矩阵
print("\n预测的评分矩阵:")
print(predictions)