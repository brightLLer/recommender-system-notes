# Recommender System(A)

> 推荐系统的作用：预测用户对一件商品的“rating”或者“prefence”

## 广义的推荐系统分类
- **简单推荐系统**（Simple recommenders）：主要基于商品的受欢迎程度进行推荐，一般来说，一件商品越受欢迎，在被推荐的时候越容易被用户所喜欢。IMDB Top 250是一个很典型的例子。
- **基于内容的推荐系统**（Content-based recommenders）：基于item metadata进行推荐，换言之，依据商品的元信息进行推荐，这种系统假设：如果用户喜欢某一特定商品，那么该用户很可能喜欢与这一特定商品类似的商品。
- **协同过滤的推荐系统**（Collaborative filtering engines）：基于过去其它用户对商品的打分或者偏好来预测当前的用户对同一商品的打分或者偏好，协同过滤不一定需要商品的元数据。

## 简单推荐系统
> 前文提到这类系统基于商品的“受欢迎程度，这种“”受欢迎程度”指的是一个特定的评价或者分数，拿IMDB Top 250的电影来说，电影的评分就可以很好的体现电影的受欢迎程度。

电影的评分可以作为推荐的依据，不过这样存在一个问题：如果电影A的平均评分是7分，评分人数是100人，而电影B的平均评分是9分，评分人数是1000人，那么推荐系统极有可能认为电影B比电影A更好，但实际上我们知道两部电影的打分人数很不平衡。这就导致了一种趋势：打分人数少的电影可能会呈现出“扭曲”或者“极端”高的评分。为了解决这一问题，提出如下一种加权评分(weighted rating/WR)的方法：
<img src="http://latex.codecogs.com/gif.latex?WR=(\frac{v}{v+m}\times R)+(\frac{m}{v+m}\times C)" />
其中：
$v$：一部电影的打分人数；$m$：人为设置的符合要求的最少打分人数；
$R$：一部电影的平均分；$C$：所有电影的平均分
$$
v=vote\_ count\ of\ a\ movie \\
R=vote\_ average\ of\ a\ movie \\
C=mean\ of\ all\ the\ moives\ '\ vote\_average
$$
我们需要深思熟虑的参数主要是$m$，一旦设定了这个值，所有打分人数少于$m$的电影就会被丢弃，剩余的都是打分人数高于$m$的电影。经常用百分位数来确定$m$的值，如90%分位数，75%分位数。以90%位数为例，一部符合要求的电影，其打分人数应该超过原来列表中90%的电影的打分人数，换言之，经过筛选的列表只留下10%的电影，如果降低到75%，则留下25%的电影。关于C和m在python里的计算，如下所示(如果电影信息的DataFrame为`meta`)：
```
import pandas as pd
metadata = pd.read_csv('./data/movies_metadata.csv', low_memory=False)

C=meta['vote_count'].mean()
m=meta['vote_count'].quantile(0.90)
```
有了$m$，就可以对原来的电影列表进行筛选了：
```
q_movies=meta.copy().loc[meta['vote_count']>=m]
# 使用copy()函数避免在原来的列表上操作
# loc函数根据index来索引，pandas的index可以是任何类型和顺序
# iloc函数则根据行号来索引，行号从0开始，只能是整数，默认index与行号相同
# ix函数则是前两者的综合
```
下面用几行代码实现第一种推荐系统:
```
def weighted_rating(x, m=m, C=C):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m) * R) + (m/(v+m) * C)
# q_movies作为参数传入weighted_rating中的x
q_moives['score']=q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)
# 推荐前面15部电影
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)
```
## 基于内容的推荐系统
> 此部分以电影的plot description/overview（情节描述）作为"内容"。比较两部电影的内容的相似性需要计算两部电影的pairwise similarity scores。

在计算相似度之前，需要为电影的overview中的**每个单词**计算TF-IDF（词频-逆向文档频率），$TF-IDF=TF\times IDF$，其中
$$
TF=\frac{该单词在该文档出现的个数}{该文档的总单词数} \\
IDF=\log(\frac{总的文档数目}{出现该单词的文档的数目}) \\
前者衡量了一个单词在一篇文档中的重要性，后者衡量了一个单词的普遍重要性。
$$
`sklearn`中有专门计算TF-IDF的模块，如下：
```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
```
有些电影的overview没有内容，读入pandas中会变成NaN，需要做如下处理：
```
meta['overview'] = meta['overview'].fillna('')
```
接下来将电影的overview转化为TF-IDF：
```
tfidf_matrix = tfidf.fit_transform(meta['overview'])
# tfidf_matrix的每一列代表字典中的一个单词，每一行代表一部电影
```
完成转换后，就可以计算电影两两之间的相似性了，相似性的计算方法有很多种类，这里采用余弦相似性，我们知道余弦函数在$[0,\frac{\pi}{2}]$上为减函数，因此两部电影(两个的tfidf向量)之间的夹角越小，余弦值越大，也意味着相似性越大。如下：
$$
\cos(x,y)=\frac{x^{T}y}{\left \|x \right \|\left \|y \right \|}
$$
实际上，`TfidfVectorizer`在计算TF-IDF之后会对向量做归一化计算，比如有两部电影overview的TF-IDF向量为$v_1$，$v_2$，那么归一化后就是：
$$
v_1^{norm}=\frac{v_1}{\left \|v_1 \right \|} \\
v_2^{norm}=\frac{v_2}{\left \|v_2 \right \|}
$$
那么，计算$v_1$，$v_2$的余弦相似性只需要计算$v_1^{norm}$，$v_2^{norm}$的线性核就可以了：
$$
(v_1^{norm})^T v_2^{norm}
=\frac{v_1^T}{\left \|v_1 \right \|}
\frac{v_2}{\left \|v_2 \right \|}=
cos(v_1,v_2)
$$
`tfidf_matrix`就是存放$v_1^{norm}$，$v_2^{norm}$这类向量，因此调用sklearn的`linear_kernel`可以节省计算时间，而不是`cosine_similarities`：
```
from sklearn.metrics.pairwies import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# cosine_sim矩阵的每一行和每一列都代表一部电影
```
接下来，我们需要思考我们的推荐系统的输入与输出是怎么样的？
> 输入：一部电影的title（标题）
输出：若干部类似电影的title（标题）

首先，在电影的title和电影的行index之间建立一个reverse map，原因在于借助电影的title可以寻找其index，再由其index到相似矩阵tfidf_matrix中找到它与其它电影的相似性，如下：
```
indices = pd.Series(meta.index, index=meta['title']).drop_duplicates()
# pd.Series()的第一个参数是column
```
然后就可以设计推荐函数了：
```
def get_recommendations(title, cosine_sim=cosine_sim):
    # 用电影的标题获取电影的index
    idx = indices[title]

    # 获取输入电影与其它电影的相似度，形成元素为(id, score)的list
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 基于相似分数进行排序
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 获取最相似的10部电影，第1部为自己与自己的相似分数，应剔除
    sim_scores = sim_scores[1:11]

    # 获取上述10部电影的id
    movie_indices = [i[0] for i in sim_scores]

    # 返回上述10部电影的title
    return metadata['title'].iloc[movie_indices]
```
> 上面的内容仅仅将电影的overview作为元数据，尝试着增加更多元数据可以改善系统的表现，比如可以试试：**the 3 top actors, the director, related genres and the movie plot keywords（前3位演员，导演，相关类别，情节关键词）**

电影的keywords（关键词）, cast（演员）和 crew（职员）信息不在上面的meta表中，要从其它数据集读取：
```
credits = pd.read_csv('./data/credits.csv')
keywords = pd.read_csv('./data/keywords.csv')

# 移除一些用不上的行，依据index
metadata = metadata.drop([19730, 29503, 35587])
# df.drop(['B', 'C'])删除列 df.drop([0, 1])删除行

# 将id的类型都转换为int，用于合并
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

#依据id合并3个数据集 
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')
```
**某一部**电影的cast, crew, keywords, genres具有如下形式：
```
cast="[{'cast_id': 14, 'character': 'Woody (voice)',...}, ...]"
crew="[{'credit_id': '52fe4284c3a36847f8024f49', 'de...}, ...]"
keywords="[{'id': 931, 'name': 'jealousy'}, {'id': 4290,...}, ...]"
genres="[{'id': 16, 'name': 'Animation'}, {'id': 35, '...}, ...]"
# 这种字符串称为"stringified"，类似于JSON字符串
```
```
from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)
# 使用python的literal_eval将stringified features解析为相关的
# python object，这里4个特征都转成了python中的list
```
接下来，提取我们最早提到的4个特征：**前3位演员，导演，相关类别，情节关键词**
```
import numpy as np

# 从crew特征列表中获取"导演"
def get_director(x):
    for i in x:
        if  i['name'] == 'director':
            return i['name']
    return np.nan

# 从cast, keywords, genres特征列表中最多返回前面3个元素，分别获取到一部电影的前3个演员，前3个情节关键词和前3个电影相关类别。
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # 多于3个则返回3个，否则全部返回
        if len(names) > 3:
            names = names[:3]
        return names
    return []

metadata['director'] = metadata['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)
```
经过处理后的表格如下：
|-|title|cast|director|keywords|genres|
|-|-|-|-|-|-|
|0|Toy Story|[Tom Hanks, Tim Allen, Don Rickles]|John Lasseter|[jealousy, toy, boy]|[Animation, Comedy, Family]|
|1|Jumanji|[Robin Williams, Jonathan Hyde, Kirsten Dunst]|Joe Johnston|[board game, disappearance, based on children'...|[Adventure, Fantasy, Family]|
|2|Grumpier Old Men|[Walter Matthau, Jack Lemmon, Ann-Margret]|Howard Deutch|[fishing, best friend, duringcreditsstinger]|[Romance, Comedy]|
接下来，要把所有特征中的大写字母变小写，并去掉空格：
```
def clean_data(x):
    # 针对上面表格的前3个特征
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # 如果导演不是np.nan
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)
```
现在，可以创建我们的metadata soup了，它将所有的特征连成一个字符串，这个字符串可以看成是一个document：
```
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) 
    + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    
metadata['soup'] = metadata.apply(create_soup, axis=1)
```
接着我们用`CountVectorizer`，注意是`CountVectorizer`而不是`TfidfVectorizer`,因为TF-IDF会强调一个单词对于某一篇特定文档而言的重要性，因此借助了idf来down-weighted，换言之，只有在这一文档内词频高，其它文档出现少(逆向文档频率高)的单词的TF-IDF才会高，但是，在我们这个例子里，演职员出现在很多部电影里是一件好事，不需要down-weighted，因此用`CountVectorizer`就可以了。
```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
```
然后，调用推荐函数进行推荐即可：
```
get_recommendations('The Dark Knight Rises', cosine_sim2)
get_recommendations('The Godfather', cosine_sim2)
```
####### 一些小建议：
- 使用上面的推荐系统获取前面30部最相似的电影，再基于这30部电影使用最早提到的简单推荐系统计算WR，排序并返回WR最高的前10部电影。
- 考虑一些其它的职员，不仅仅是导演
- 加大导演的权重：例如可以让导演的名字在soup中重复几次
参考网址：[https://www.datacamp.com/community/tutorials/recommender-systems-python](https://www.datacamp.com/community/tutorials/recommender-systems-python)
