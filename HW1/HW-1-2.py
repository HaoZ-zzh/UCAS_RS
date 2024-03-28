import numpy as np
# import openpyxl
import pandas as pd

# 用户评分矩阵
ratings = np.array([
    [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],  # 用户A的评分
    [3.0, np.nan, 1.5, 5.0, 3.0, 3.5],  # 用户B的评分
    [2.5, 3.5, np.nan, 3.5, 4.0, np.nan],  # 用户C的评分
    [3.5, 2.0, 4.5, np.nan, 3.5, 2.0],  # 用户D的评分
    [3.0, 4.0, 2.0, 3.0, 3.0, 2.0],  # 用户E的评分
    [4.5, 1.5, 3.0, 5.0, 3.5, np.nan]  # 用户F的评分
])

user_means = np.nanmean(ratings, axis=1)

# 定义Cosine相似度函数
def cosine_similarity(item_i, item_j):
    vector_i = ratings[:, item_i]
    vector_j = ratings[:, item_j]
    
    # 找出同时对这两个物品进行评分的用户
    both_rated_users = ~np.isnan(vector_i) & ~np.isnan(vector_j)
    rated_users_indices = np.where(both_rated_users)[0]
    v1 = vector_i[rated_users_indices]
    v2 = vector_j[rated_users_indices]
    dot_product = np.sum(v1 * v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# 计算调整后的余弦相似度的函数
def adjusted_cosine_similarity(item_i, item_j):
    sum_dot_product = 0
    sum_squared_diff_user_mean_i = 0
    sum_squared_diff_user_mean_j = 0
    non_nan_user_count = 0
    
    # 遍历所有用户
    for user_id in range(ratings.shape[0]):
        if not np.isnan(ratings[user_id, item_i]) and not np.isnan(ratings[user_id, item_j]):
            # 计算评分偏差
            deviation_i = ratings[user_id, item_i] - user_means[user_id]
            deviation_j = ratings[user_id, item_j] - user_means[user_id]
            
            # 累加分子和分母的和
            sum_dot_product += deviation_i * deviation_j
            sum_squared_diff_user_mean_i += deviation_i ** 2
            sum_squared_diff_user_mean_j += deviation_j ** 2
            
            non_nan_user_count += 1
    
    # 计算分母的平方根
    denominator_i = np.sqrt(sum_squared_diff_user_mean_i / non_nan_user_count) if non_nan_user_count else 0
    denominator_j = np.sqrt(sum_squared_diff_user_mean_j / non_nan_user_count) if non_nan_user_count else 0
    
    # 避免除以零
    if denominator_i == 0 or denominator_j == 0:
        return 0
    
    # 计算调整后的余弦相似度
    return sum_dot_product / (denominator_i * denominator_j)

# 计算整个物品相似度矩阵的函数
def calculate_similarity_matrix(ratings, similarity_func=cosine_similarity):
    num_items = ratings.shape[1]
    item_similarity = np.zeros((num_items, num_items))
    
    for i in range(num_items):
        for j in range(i+1, num_items):  # 避免重复和自我比较
            
            # 计算这两个向量的相似度
            item_similarity[i][j] = similarity_func(i, j)
            item_similarity[j][i] = item_similarity[i][j]  # 确保对称性
    
    return item_similarity

# 预测用户对物品的评分的函数
# 预测整个评分矩阵的函数
def predict_rating_matrix(ratings, item_similarity_matrix):
    num_users, num_items = ratings.shape
    predicted_ratings = np.nan * np.ones((num_users, num_items))
    
    for user_id in range(num_users):
        for item_id in range(num_items):
            if np.isnan(ratings[user_id, item_id]):
                # 计算用户已评分的物品的评分
                rated_items = ratings[user_id, :]
                # print(rated_items)
                rated_items_indices = np.where(~np.isnan(rated_items))[0]
                # print(rated_items_indices)
                
                # 计算预测评分的分子和分母
                numerator = 0
                denominator = 0
                
                for rated_item_index in rated_items_indices:
                    similarity = item_similarity_matrix[rated_item_index, item_id]
                    rating = ratings[user_id, rated_item_index]
                    numerator += similarity * rating
                    denominator += similarity
                
                if denominator != 0:
                    predicted_ratings[user_id, item_id] = numerator / denominator
    
    return predicted_ratings



# 计算物品相似度矩阵
item_similarity_matrix = calculate_similarity_matrix(ratings, similarity_func=cosine_similarity)
# item_similarity_matrix = calculate_similarity_matrix(ratings, similarity_func=adjusted_cosine_similarity)
# print("物品相似度矩阵:")
# print(item_similarity_matrix)

predicted_ratings_matrix = predict_rating_matrix(ratings, item_similarity_matrix)

# 打印预测的评分矩阵
# print("Predicted rating matrix:")
# print(predicted_ratings_matrix)


# 为物品和用户创建标签列表
item_labels = [f'电影{i+1}' for i in range(6)]  # 假设有6部电影
user_labels = ['用户A', '用户B', '用户C', '用户D', '用户E', '用户F']  # 用户标签

# 创建一个Excel文件写入器
with pd.ExcelWriter('hw1-2.xlsx', engine='openpyxl') as writer:
    # 将物品相似度矩阵转换为DataFrame并保存到Excel
    item_similarity_df = pd.DataFrame(item_similarity_matrix, index=item_labels, columns=item_labels)
    item_similarity_df.to_excel(writer, sheet_name='Item_Similarity', index=True, header=True)
    
    # 在物品相似度矩阵后添加三行空白行
    for _ in range(3):
        writer.sheets['Item_Similarity'].append([np.nan] * len(item_labels))

    # 将预测评分矩阵转换为DataFrame并保存到Excel
    predicted_ratings_df = pd.DataFrame(predicted_ratings_matrix, index=user_labels, columns=item_labels)
    predicted_ratings_df.to_excel(writer, sheet_name='Predicted_Rating', index=True, header=True)

print(f"Excel files have been saved to 'output_with_spaces.xlsx' with item similarity matrix and predicted rating matrix separated by three blank lines.")