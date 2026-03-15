# UK Parliament Debate NLP Analysis

本项目是一个综合性的自然语言处理（NLP）分析系统，专门针对英国国会辩论数据进行深度分析和可视化。项目集成了主题建模、词向量聚类、情感分析和文本复杂度分析等多种NLP技术。

## 项目概述

该项目基于 UK Parliament (ParlVote) 辩论数据集，通过多种先进的NLP方法分析国会议员的辩论内容。主要分析方向包括：

- **主题建模**：使用 BERTopic 和 LDA 识别辩论中的主要话题
- **词向量聚类**：通过 Word2Vec 结合 K-Means 和 CLARA 算法进行文本聚类
- **情感分析**：利用 RoBERTa 模型分析辩论的情感倾向
- **文本复杂度**：评估议员发言的复杂度水平
- **可视化分析**：生成词云、热力图、时间序列等多种可视化图表

## 数据集

### ParlVote 数据集

数据集来源于英国国会辩论记录，包含以下主要字段：

| 字段名 | 说明 |
|--------|------|
| `debate_id` | 辩论唯一标识符（前4位为年份） |
| `motion_party` | 提出议案的政党 |
| `debate_title` | 辩论标题 |
| `motion_text` | 议案文本内容 |
| `party` | 发言议员所属政党 |
| `speech` | 议员发言内容 |

### 涵盖的政党

- Conservative（保守党）
- Labour（工党）
- Liberal-Democrat（自由民主党）
- Scottish National Party（苏格兰民族党）
- DUP（民主统一党）
- UUP（阿尔斯特联合党）
- Green（绿党）
- Alliance（联盟党）
- Plaid Cymru（威尔士民族党）
- UKIP（英国独立党）
- Independent（独立议员）
- 等等

## 项目结构

```
NLP-FinalHW/
├── bertopic_main.py              # BERTopic 主题建模主程序
├── lda_main.py                   # LDA 主题建模主程序
├── word2vec_kmeans_clustering.py # Word2Vec + K-Means 聚类
├── word2vec_clara_clustering.py  # Word2Vec + CLARA 聚类
├── evaluate_clustering.py        # 聚类评估脚本
├── generate_wordcloud.py         # 政党词云生成
├── generate_timeline_wordcloud.py # 年度词云时间线生成
├── senti_complexity_roberta_base.py # RoBERTa 情感与复杂度分析

├── corpus/                       # 原始数据目录
│   └── ParlVote_concat.csv       # 合并后的辩论数据
├── models/                       # 保存的模型
├── stopwords/                    # 停用词表
│   ├── bert_stopwords.txt        # BERTopic 停用词
│   ├── lda_stopwords.txt         # LDA 停用词
│   ├── w2v_stopwords.txt         # Word2Vec 停用词
│   └── w2v_stopwords_clara.txt  # CLARA 聚类停用词

├── bertopic_analysis_8/          # BERTopic 分析结果（8主题）
├── bertopic_analysis_None/       # BERTopic 分析结果（自动聚类）
├── lda_analysis/                 # LDA 分析结果
├── clustering_analysis_9/        # K-Means 聚类结果（9类）
├── clustering_analysis_10/        # K-Means 聚类结果（10类）
├── clustering_analysis_CLARA/    # CLARA 聚类结果
├── topic2Vec_analysis/           # Topic2Vec 分析结果
├── senti_complexity_roberta/     # 情感复杂度分析结果
├── wordcloud/                    # 政党词云
└── timeline_wordcloud/           # 年度词云时间线
```

## 环境配置

### 依赖要求

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
gensim>=4.1.0
scikit-learn>=1.0.0
joblib>=1.1.0
tqdm>=4.62.0
nltk>=3.6.0
bertopic>=0.15.0
sentence-transformers>=2.2.0
textstat>=0.7.3
scipy>=1.7.0
torch>=2.0.0
transformers>=4.30.0
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. BERTopic 主题建模

```bash
python bertopic_main.py
```

**功能说明**：
- 自动进行文本预处理（清洗、分词、词形还原、停用词过滤）
- 使用 BERTopic 进行主题建模
- 生成主题-政党分布热力图
- 生成主题-年份分布热力图
- 计算模型评估指标（一致性、多样性、困惑度）

**输出文件**：
- `bertopic_topic_results.txt` - 主题关键词
- `bertopic_party_topic_distribution.png` - 政党主题分布图
- `bertopic_topic_party_heatmap.png` - 主题-政党热力图
- `bertopic_year_topic_distribution.png` - 年度主题分布图
- `bertopic_topic_visualization.html` - 交互式主题可视化

### 2. LDA 主题建模

```bash
python lda_main.py
```

**功能说明**：
- 使用 LDA (Latent Dirichlet Allocation) 进行主题建模
- 多线程并行处理
- 主题一致性评估
- 困惑度计算

**输出文件**：
- `lda_topic_results.txt` - LDA 主题结果
- `lda_party_topic_distribution.png` - 政党主题分布
- `lda_topic_year_heatmap.png` - 主题年份热力图

### 3. Word2Vec 聚类分析

#### K-Means 聚类

```bash
python word2vec_kmeans_clustering.py
```

#### CLARA 聚类

```bash
python word2vec_clara_clustering.py
```

**功能说明**：
- 训练 Word2Vec 词向量模型
- 使用 K-Means 或 CLARA 算法进行聚类
- 生成词向量可视化（t-SNE/PCA）
- 评估聚类质量（轮廓系数、Calinski-Harabasz 指数）

**输出文件**：
- `word2vec_lemmatized_figure.png` - 词向量可视化图
- `cluster_sentiment_stats.csv` - 聚类情感统计
- `elbow_method_curve_tfidf.png` - 肘部法则曲线

### 4. 聚类评估

```bash
python evaluate_clustering.py
```

**功能说明**：
- 综合评估不同聚类数的效果
- 生成轮廓系数趋势图
- 输出最优聚类数建议

### 5. 词云生成

#### 政党词云

```bash
python generate_wordcloud.py
```

#### 年度词云时间线

```bash
python generate_timeline_wordcloud.py
```

**输出文件**：
- `wordcloud/{party}_wordcloud.png` - 各政党词云
- `timeline_wordcloud/{year}_wordcloud.png` - 年度词云
- `timeline_wordcloud/yearly_wordcloud_timeline.png` - 年度词云合集

### 6. 情感与复杂度分析

```bash
python senti_complexity_roberta_base.py
```

**功能说明**：
- 使用 RoBERTa 模型进行情感分析
- 计算文本复杂度指标（可读性分数）
- 按政党、年份分析情感趋势
- 生成散点图、气泡图、时间序列图

**输出文件**：
- `senti_complexity_roberta/raw/` - 原始情感分析结果
- `senti_complexity_roberta/z-score/` - Z-Score 标准化结果
- `time_series_sentiment_yearly.png` - 年度情感趋势
- `time_series_complexity_yearly.png` - 年度复杂度趋势

## 分析结果示例

### 主题建模

项目生成了多种主题模型分析结果，包括：
- 8主题和自动聚类两种模式的 BERTopic 结果
- 8主题的 LDA 结果
- Topic2Vec 主题相似度分析

### 情感分析

通过 RoBERTa 模型分析发现：
- 各政党发言的情感倾向存在显著差异
- 随时间推移，情感和复杂度呈现特定趋势

### 词云可视化

- 按政党生成专属词云，展示各党关注焦点
- 年度词云时间线展示话题演变

## 核心算法说明

### BERTopic

基于 Transformer 的主题建模框架，结合 BERT 词向量和聚类算法，能够自动发现语义连贯的主题。

### LDA

经典的概率主题模型，通过隐变量发现文档集合中的主题结构。

### Word2Vec

词嵌入技术，将词语映射到低维向量空间，捕捉词语之间的语义关系。

### CLARA (Clustering LARge Applications)

针对大规模数据的聚类算法，是 PAM (Partitioning Around Medoids) 的改进版本。

### RoBERTa

强大的预训练语言模型，用于情感分析和文本分类任务。

## 注意事项

1. **数据路径**：确保 `corpus/ParlVote_concat.csv` 存在于项目根目录
2. **内存要求**：部分分析（如 BERTopic）需要较大内存，建议 16GB+ RAM
3. **多核利用**：程序会自动检测 CPU 核心数并并行处理
4. **模型保存**：训练好的模型会保存在 `models/` 目录下

## 许可证

本项目仅供学术研究使用。
