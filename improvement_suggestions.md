# 文本处理任务改进与扩展建议

## 一、当前已完成的任务

1. **主题挖掘**：使用LDA模型对议会投票文本进行主题分析
2. **文本聚类**：
   - K-Means聚类
   - 层次聚类
   - 基于词频特征和PCA降维的可视化

## 二、进一步改进与扩展方向

### 1. 文本表示方法改进

#### 1.1 词向量表示
- **TF-IDF加权**：替代当前的CountVectorizer，使用TfidfVectorizer计算词频权重
- **Word2Vec/GloVe词向量**：使用预训练词向量或在语料上训练自定义词向量
- **Doc2Vec文档向量**：直接生成文档级别的向量表示
- **BERT嵌入**：使用预训练BERT模型生成上下文相关的文档嵌入

#### 1.2 特征选择优化
- **互信息（MI）**：基于互信息选择与党派相关性高的词语
- **卡方检验**：过滤掉与类别相关性低的特征
- **L1正则化**：使用LogisticRegression的L1正则化进行特征选择
- **主题特征**：将LDA主题分布作为额外特征

### 2. 聚类算法改进

#### 2.1 算法选择扩展
- **DBSCAN**：基于密度的聚类，适合发现任意形状的簇
- **谱聚类**：适合高维数据的聚类
- **GMM（高斯混合模型）**：软聚类，给出每个样本属于不同簇的概率

#### 2.2 参数优化
- **K-Means**：使用肘部法则+轮廓系数结合确定最佳聚类数
- **层次聚类**：尝试不同的链接方法（complete, average, single）
- **DBSCAN**：优化eps和min_samples参数

#### 2.3 特征降维改进
- **t-SNE**：比PCA更适合可视化高维数据的非线性结构
- **UMAP**：保留局部和全局结构的降维方法
- **LDA主题分布**：直接使用主题分布作为低维特征

### 3. 主题模型改进

#### 3.1 模型扩展
- **NMF（非负矩阵分解）**：获得更易解释的主题
- **LDA参数优化**：使用GridSearchCV优化alpha和beta参数
- **动态主题模型**：如果数据包含时间信息，分析主题随时间的演化
- **BERTopic**：结合BERT嵌入和聚类的现代主题模型

#### 3.2 主题评估
- **困惑度（Perplexity）**：评估主题模型的泛化能力
- **一致性分数（Coherence Score）**：评估主题的可解释性

### 4. 添加序列标注任务

#### 4.1 命名实体识别（NER）
- 识别文本中的实体：人名、机构、法案名称等
- 使用模型：
  - CRF（条件随机场）
  - BiLSTM-CRF
  - BERT-CRF

#### 4.2 情感分析
- 分析议会文本的情感倾向
- 使用模型：
  - 传统机器学习：SVM+TF-IDF
  - 深度学习：LSTM、BERT

### 5. 深度神经网络方法

#### 5.1 聚类任务
- **自编码器+K-Means**：使用自编码器学习低维特征，再进行K-Means聚类
- **变分自编码器（VAE）**：生成高质量的低维嵌入用于聚类
- **图神经网络（GNN）**：将文本视为图结构，进行图聚类

#### 5.2 主题模型
- **Neural LDA**：使用神经网络实现的LDA
- **GPT-based主题模型**：利用大语言模型生成主题

## 三、具体实现建议

### 1. 实现TF-IDF+K-Means改进
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 使用TF-IDF替代CountVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words=list(all_stopwords))
word_freq_matrix = vectorizer.fit_transform(party_texts['processed_text'])
```

### 2. 实现Word2Vec词向量
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# 准备词向量训练数据
tokenized_texts = [word_tokenize(text) for text in df['processed_text']]

# 训练Word2Vec模型
w2v_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=5, workers=4)

# 生成文档向量（平均词向量）
def get_doc_vector(text, model):
    tokens = word_tokenize(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

df['doc_vector'] = df['processed_text'].apply(lambda x: get_doc_vector(x, w2v_model))
```

### 3. 实现BERTopic主题模型
```python
from bertopic import BERTopic

# 使用BERTopic进行主题建模
topic_model = BERTopic(language="english", calculate_probabilities=True)
topics, probs = topic_model.fit_transform(df['processed_text'])

# 保存主题结果
topic_info = topic_model.get_topic_info()
topic_info.to_csv('motion_lda/bertopic_topics.csv', index=False)
```

### 4. 实现t-SNE可视化
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
word_freq_tsne = tsne.fit_transform(word_freq_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(word_freq_tsne[:, 0], word_freq_tsne[:, 1], c=kmeans_labels, cmap='viridis')
for i, party in enumerate(party_texts['motion_party']):
    plt.annotate(party, (word_freq_tsne[i, 0], word_freq_tsne[i, 1]))
plt.title('t-SNE可视化（K-Means聚类）')
plt.savefig('motion_cluster/tsne_visualization.png')
```

### 5. 实现自编码器聚类
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 构建自编码器
input_dim = word_freq_scaled.shape[1]
encoding_dim = 50

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(word_freq_scaled, word_freq_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# 使用编码器生成低维特征
encoded_features = encoder.predict(word_freq_scaled)

# 在低维特征上进行K-Means聚类
kmeans_ae = KMeans(n_clusters=best_k, random_state=42)
kmeans_ae_labels = kmeans_ae.fit_predict(encoded_features)
```

## 四、实验设计建议

### 1. 对比实验
| 文本表示 | 聚类算法 | 降维方法 | 轮廓系数 |
|---------|---------|---------|---------|
| TF-IDF  | K-Means | PCA     | ?       |
| TF-IDF  | K-Means | t-SNE   | ?       |
| Word2Vec| K-Means | 无      | ?       |
| Doc2Vec | 层次聚类| PCA     | ?       |
| BERT嵌入| DBSCAN  | UMAP    | ?       |

### 2. 主题模型对比
| 模型 | 主题数 | 困惑度 | 一致性分数 | 可解释性 |
|------|--------|--------|------------|----------|
| LDA  | 5      | ?      | ?          | ?        |
| LDA  | 10     | ?      | ?          | ?        |
| NMF  | 5      | -      | ?          | ?        |
| BERTopic | 自动 | -      | ?          | ?        |

## 五、预期效果

1. **提高聚类质量**：通过更好的文本表示和参数优化，获得更高的轮廓系数
2. **增强主题可解释性**：使用更先进的主题模型，获得更清晰的主题
3. **丰富分析维度**：添加序列标注任务，从多个角度分析文本
4. **提升可视化效果**：使用t-SNE等方法，获得更直观的聚类结果可视化
5. **加深对数据的理解**：通过对比实验，了解不同方法的优缺点

## 六、代码结构优化

1. **模块化设计**：将文本预处理、特征提取、模型训练、结果可视化分离为独立模块
2. **配置文件**：使用JSON/YAML配置文件管理参数，方便进行参数调优
3. **日志记录**：添加日志记录，方便跟踪实验过程
4. **结果自动生成**：自动生成实验报告，包括表格、图表和分析结论

通过以上改进和扩展，可以更全面地完成作业要求，并深入探索不同文本处理方法的效果。