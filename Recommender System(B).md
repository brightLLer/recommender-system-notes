# Recommender System(B)
> 接着A部分继续补充，B部分主要讨论基于协同过滤(Collaborative filtering)的推荐系统。协同过滤推荐的基础是：users对items的attitude（态度）。这点与基于内容推荐不同：基于item的attribute（元数据）。

## 协同过滤的种类
- **Memory-Based Collaborative Filtering**：基于余弦相似性的计算
- **Model-Based Collaborative Filtering**：基于矩阵分解

> 这部分实验中将要使用到的数据集是MovieLens，它是实现和测试推荐引擎最常用的数据集，它包括了943位users对1682部电影的100k个ratings。

```
import numpy as np
import pandas as pd

# 读取ml-100k数据集
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# 分割训练集和测试集
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)
```
## Memory-Based Collaborative Filtering
这种协同过滤又分为两种，user-item filtering和item-item filtering。其中，前者接受一个**特定的用户作为输入**，从而基于打分的相似性找到与该用户相似的用户，然后将商品推荐给那些喜欢的相似的用户；后者接受**一件特定的商品作为输入**，然后找到喜欢这件商品的用户，推荐一些相似商品给这些用户。下面是两句简记：
> Item-Item Collaborative Filtering: “Users who liked this item also liked …”
User-Item Collaborative Filtering: “Users who are similar to you also liked …”

针对训练集和测试集，一共需要创建两个943 * 1682的user-item矩阵，训练集矩阵保留75%的打分，测试集保留25%的打分，下面是一个user-item矩阵的示例：
![](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/figures/BLOG_CCA_8.png)
有了user-item矩阵之后就可以计算相似分，从而构建相似矩阵了。item-item协同过滤通过观测所有用户对成对的item的打分进行相似性计算，如下图所示：
![](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/figures/BLOG_CCA_10.png)
相反，user-item协同过滤通过观测成对的用户对所用item的打分进行相似性计算，如下图所示：
![](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/figures/BLOG_CCA_11.png)
设user-item矩阵为$x$，那么user-item协同过滤中，用户$k$和用户$a$之间的相似性计算如下：
$$
s_u^{cos}(u_k,u_a)=\frac{u_k\cdot u_a}{\left \|u_k \right \|\left \|u_a \right \|}=\frac{\sum{x_{a,m}x_{k,m}}}{\sqrt{\sum{x_{a,m}^2}\sum{x_{k,m}^2}}}
$$
而在item-item协同过滤中，商品$m$和商品$a$之间的相似性计算为：
$$
s_u^{cos}(i_m,i_a)=\frac{i_m\cdot i_a}{\left \|i_m \right \|\left \|i_a \right \|}=\frac{\sum{x_{a,m}x_{a,b}}}{\sqrt{\sum{x_{a,m}^2}\sum{x_{a,b}^2}}}
$$
下面代码首先创建training和testing的user-item矩阵：
```
# itertuples()形成了一个元素为(user_id,item_id,rating,timestamp)的list
# line[1]：user_id; line[2]:item_id; line[3]:rating
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
```
然后，使用sklearn的pairwise_distance来计算余弦相似性：
```
# pairwise_distance(X, Y=None)的返回值是一个距离矩阵D，D_{ij}表示
# X的第i个行向量与第j个行向量之间的距离,如果Y不为None，则表示X的第i
# 行向量和Y的第j个行向量之间的距离
from sklearn.metrics.pairwise import pairwise_distances
#shape=(943,943)
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
#shape=(1682, 1682)
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
```
接下来进入预测环节，首先是user-based CF，其预测方法如下：
$$
\hat{x}_{k,m}=\overline{x}_{u_k}+\frac{\sum_{u_a}{sim_{u}(u_k,u_a)(x_{a,m}-\overline{x}_{u_a})}}{\sum_{u_a}|sim_{u}(u_k,u_a)|}
$$
> **公式理解：**公式左边$\hat{x}_{k,m}$表示用户$k$对商品$m$的评分，即待预测项，公式右边，首先计算user-item矩阵第$k$列的平均结果$\overline{x}_{u_k}$，它表示用户$u_k$对所有商品的预测平均分，它起到校正的作用，因为有些用户可能趋向于打高分或低分，接着，所加上的后面这一项中，我们先看分子，先不要看求和符号，$sim_{u}(u_k,u_a)$是$u_k$和$u_a$的相似分，不要忘记我们使用user-based CF，用户与用户之间的相似性要涉及，这一项可以看成是一个权重，$x_{a,m}-\overline{x}_{u_a}$是去中心化后用户$a$对商品$m$的评分，分母用于归一化，让打分位于1到5之间。
**一个数学小技巧：**涉及到两个矩阵元素的乘积再求和，要迅速联想到矩阵乘法，比如这里涉及到了$sim_{u}(u_k,u_a)$和$x_{a,m}$的乘积再求和，实际上就是$sim$和$x$两个矩阵的乘法，结合两个矩阵的形状再思考谁在前面谁在后面，比如假设user-item矩阵为$R^{m\times n}$，则user-sim矩阵为$R^{m\times m}$，那么矩阵乘法应该是类似于$sim\cdot x$的形式。

接着，是item-based CF的预测方法，这种方法不需要加入用户校正项：
$$
\hat{x}_{k,m}=\frac{\sum_{i_b}{sim_{u}(i_m,i_b)(x_{k,b})}}{\sum_{i_b}|sim_{u}(i_m,i_b)|}
$$
> 公式理解：item-based CF看的是同一位用户对不同但相似的商品k和b打分的相似性，公式左边是用户k对商品$m$的打分，因此右边要考察用户$k$对商品b的打分,$sim(i_m,i_b)$依然可以看成一个权重。若设user-item矩阵为$R^{m\times n}$，则item-sim为$R^{n\times n}$
```
def predict(ratings, similarity, type='user'):
    if type == 'user':
        # 可以使用keep_dims=True，就不需要np.newaxis
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        # (m * m) * (m * n) / (1, m) broadcasting
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
    # 同样可以使用keep_dims=True，分子就不需要再外面再套一层np.array()形成二维数组了
    # (m * n) * (n * n) / (1, n) broadcasting
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
```
下面就是预测了：
```
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')
```
接下来是使用RMSE进行评估：
$$
RMSE=\sqrt{\frac{1}{N}\sum_i^N{(x_i-\hat{x_i})^2}}
$$
```
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
```
> ndarray.nonzero()：返回数组中非零元素的索引。
比如：
\>\>\>import numpy as np
\>\>\>arr = np.array([[1, 0], [0, 2]])
\>\>\>arr.nonzero()
(array([0, 1], dtype=int64), array([0, 1], dtype=int64))
返回的两个数组分别代表非零元素的行索引数组和列索引数组，因此，两个非零元素的位置为(0,0)和(1,1)。

再然后是调用：
```
rmse(user_prediction, test_data_matrix)
```
**总结：memory-based CF 推荐系统虽然易于实现，但它不能解决冷启动问题，一旦有新的user或者item加入系统，它不能很好地处理。model-based CF可以更好比前者更好地处理大规模和高稀疏度的矩阵，但也无法解决冷启动问题。**
## Model-Based Collaborative Filtering
> 这种协同过滤基于矩阵分解(MF)，可以当成一个无监督学习问题（潜变量分解和降维），在数据的规模和稀疏程度的处理上比memory-based CF更好。Model-Based CF可以从已知打分中学习到用户潜在的偏好和商品潜在的属性，通过对用户和商品的潜在特征执行矩阵的乘法运算来预测用户对未知商品的打分。如果有一个高稀疏又高维的矩阵，可以使用MF将该矩阵分解为若干low rank矩阵相乘。
```
# 计算稀疏等级
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')
```
再用一个直观的例子解释一下model-based CF,MovieLens数据集有这样一些特征(user id, age, location, gender, movie id, director, actor, language, year, rating)，通过MF，模型学习到的重要用户特征是age group (under 10, 10-18, 18-30, 30-90), location和 gender，而重要的商品（电影特征）是decade, director和actor，decade是原来的特征所没有的，它就是一个潜在的特征，模型可以自己学习到。model-based CF需要使用的特征会比较多，如果特征太少，就难以学习到潜在的特征。
> 小注：Models that use both ratings and content features are called Hybrid Recommender Systems，they are capable to address the cold-start problem better since if you don't have any ratings for a user or an item you could use the metadata from the user or item to make a prediction.

一种常见的MF方法是SVD，原始的user-item矩阵$X$被分解为3个矩阵：
$$
X=USV^T \\
U:m\times r的正交矩阵，每一行代表每个用户的（隐藏）特征 \\
S:r\times r的对角矩阵，对角线从大到小排列着奇异值 \\
V^T:r\times n的正交矩阵，V\in R^{n\times r}，V的每一行代表每件商品的（隐藏）特征
$$
下图展现了模型进行MF（学习）的过程：
![](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/figures/BLOG_CCA_5.png)
然后是预测，预测用了约等于号：
![](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/figures/BLOG_CCA_4.png)
```
from scipy.sparse.linalg import svds

#从训练矩阵中获取主要的svd components，取k=20
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))
```
**总结：SVD的缺点是计算慢，代价昂贵，处理不当会引起过拟合：More recent work minimizes the squared error by applying alternating least square or stochastic gradient descent and uses regularization terms to prevent overfitting.**
参考网址：[https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html)
