import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import os
from time import time
from collections import Counter

# NLP & 机器学习库
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# 并行处理库
from joblib import Parallel, delayed
from tqdm import tqdm

# 设置绘图风格
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
except:
    pass

# ==========================================
# 0. 设定核心数
# ==========================================
NUM_CORES = max(1, multiprocessing.cpu_count() - 2)
print(f"★ 已设定使用 {NUM_CORES} 个线程进行并行加速")

# ==========================================
# 辅助函数
# ==========================================

def get_wordnet_pos(tag):
    """
    将NLTK的POS标签转换为WordNet的POS标签
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # 默认使用名词

def load_stopwords(filepath='stopwords/w2v_stopwords_clara.txt'):
    stopwords = set()
    if os.path.exists(filepath):
        print(f"正在加载停用词表: {filepath} ...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f if line.strip())
            print(f"成功加载 {len(stopwords)} 个停用词。")
        except Exception as e:
            print(f"加载停用词失败: {e}")
    return stopwords

# ★ 修改：添加 use_lemmatization 参数并实现词形还原逻辑
def preprocess_wrapper(text, stopwords_set=None, use_lemmatization=True):
    """
    预处理流程：分词 -> 去停用词 -> 词形还原
    """
    if pd.isna(text):
        return []
    
    # 1. 分词 (Gensim 的 simple_preprocess 会自动转小写并去标点)
    tokens = simple_preprocess(str(text))
    
    # 2. 先去停用词 (此时 token 是完整的，能匹配上 stopwords 里的 'government')
    if stopwords_set:
        tokens = [t for t in tokens if t not in stopwords_set]
    
    # 3. 最后对剩下的词进行词形还原
    if use_lemmatization:
        # 在函数内初始化，防止多进程冲突
        lemmatizer = WordNetLemmatizer()
        # 简化词形还原（不使用POS标签，避免多进程序列化问题）
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        
    return tokens

def get_cluster_keywords(texts, stop_words_list=None, top_n=5):
    """
    提取聚类关键词 (TF-IDF)
    自动处理停用词的词形还原，以匹配输入文本
    """
    if not texts: return "N/A"
    
    # 初始化词形还原器
    lemmatizer = WordNetLemmatizer()
    
    # 1. 准备基础停用词
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    base_stop_words = list(ENGLISH_STOP_WORDS)
    
    # 2. 合并用户自定义停用词
    combined_stopwords = base_stop_words
    if stop_words_list:
        combined_stopwords.extend(list(stop_words_list))
        
    # 3. ★ 关键步骤：对停用词表也进行词形还原
    # 这样停用词表里就同时有了 'government' 和 'government'（词形还原结果相同）
    # 无论输入文本是原始的还是词形还原的，都能被过滤
    lemmatized_stopwords = []
    for w in combined_stopwords:
        # 简单词形还原（不使用POS标签，因为停用词主要是常见词）
        lemma = lemmatizer.lemmatize(w)
        lemmatized_stopwords.append(lemma)
    
    # 合并 原词 + 词形还原词 (去重)
    final_stop_words = list(set(combined_stopwords + lemmatized_stopwords))

    # 4. 运行 TF-IDF
    tfidf = TfidfVectorizer(stop_words=final_stop_words, max_features=1000)
    
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = np.array(tfidf.get_feature_names_out())
        avg_weights = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = avg_weights.argsort()[::-1][:top_n]
        return ", ".join(feature_names[top_indices])
    except ValueError:
        return "N/A"

def get_tfidf_weighted_vector(doc_tokens, w2v_model, idf_dict, vector_size):
    """
    获取文档向量：计算 TF-IDF 加权平均值
    """
    valid_tokens = [word for word in doc_tokens if word in w2v_model.wv.key_to_index and word in idf_dict]
    
    if not valid_tokens:
        return np.zeros(vector_size)
    
    tf_counter = Counter(valid_tokens)
    total_tokens = len(valid_tokens)
    
    weighted_sum = np.zeros(vector_size)
    total_weight = 0.0
    
    for word, count in tf_counter.items():
        vec = w2v_model.wv[word]
        tf = count / total_tokens
        idf = idf_dict[word]
        weight = tf * idf
        
        weighted_sum += vec * weight
        total_weight += weight
        
    if total_weight == 0:
        return np.zeros(vector_size)
        
    return weighted_sum / total_weight


def main():
    start_total = time()

    # ==========================================
    # 1. 数据读取与预处理
    # ==========================================
    csv_path = 'corpus/ParlVote_concat.csv' 
    print(f"正在读取数据: {csv_path} ...")
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=['motion_text', 'speech'])
        else:
            raise FileNotFoundError("CSV file not found")
    except Exception:
        print(f"读取 CSV 失败，使用模拟数据...")
        topics = ['economy', 'health', 'war', 'education', 'environment']
        data_text = [f"Motion about {topics[i%5]} and policy development" for i in range(2000)]
        df = pd.DataFrame({'motion_text': data_text, 'speech': data_text})

    all_texts = pd.concat([df['motion_text'], df['speech']]).dropna().astype(str)
    unique_texts = all_texts.unique()
    print(f"唯一文档数量: {len(unique_texts)}")

    # 加载停用词 (建议把 'government', 'people', 'house' 等高频政治词加入这个txt文件)
    stopwords_set = load_stopwords('stopwords/w2v_stopwords.txt')

    # 设定是否开启词形还原
    USE_LEMMATIZATION = True 
    
    # 并行处理：先去停用词，再词形还原
    tokenized_docs = Parallel(n_jobs=NUM_CORES)(
        delayed(preprocess_wrapper)(text, stopwords_set, USE_LEMMATIZATION) for text in tqdm(unique_texts, desc="Tokenizing")
    )
    
    valid_docs = []
    valid_indices = []
    for i, doc in enumerate(tokenized_docs):
        if len(doc) > 0:
            valid_docs.append(doc)
            valid_indices.append(i)
            
    unique_texts = unique_texts[valid_indices]
    print(f"有效文档数量: {len(valid_docs)}")

    # ==========================================
    # 3. Word2Vec 模型加载或训练
    # ==========================================
    save_dir = 'clustering_analysis_CLARA'
    os.makedirs(save_dir, exist_ok=True)
    # 修改模型文件名以区分是否使用了词干化
    model_name = 'word2vec_clara.model'
    model_path = os.path.join("w2v_models", model_name)
    
    model = None
    if os.path.exists(model_path):
        print(f"\n★ 检测到已有模型: {model_path}")
        try:
            loaded_model = Word2Vec.load(model_path)
            print(">>> 模型加载成功！")
            model = loaded_model
        except Exception as e:
            print(f">>> 模型加载出错 ({e})，将重新训练。")
    
    if model is None:
        print(f"\n正在训练新 Word2Vec 模型 (workers={NUM_CORES}, epochs=40)...")
        model = Word2Vec(sentences=valid_docs, vector_size=100, window=5, min_count=2, workers=NUM_CORES, epochs=40, seed=42, sg=1)
        model.save(model_path)
        print("模型训练并保存完毕。")

    # ==========================================
    # 4. 生成文档向量 (直接使用Word2Vec平均向量)
    # ==========================================
    print("\n★ 正在生成 Word2Vec 平均文档向量...")
    
    corpus_as_strings = [" ".join(doc) for doc in valid_docs]
    
    vector_size = model.vector_size
    doc_vectors = []
    
    for doc in tqdm(valid_docs, desc="Vectorizing"):
        # 直接计算文档中所有词向量的平均值
        valid_words = [word for word in doc if word in model.wv.key_to_index]
        if valid_words:
            vec = np.mean([model.wv[word] for word in valid_words], axis=0)
        else:
            vec = np.zeros(vector_size)
        doc_vectors.append(vec)
        
    doc_vectors = np.array(doc_vectors)
    doc_vectors_norm = normalize(doc_vectors)

    # ==========================================
    # 5. CLARA 聚类
    # ==========================================
    print(f"\n正在使用 CLARA 运行聚类...")
    
    # 设置CLARA参数
    num_clusters = 10  # 设定聚类数
    samples = 5  # 样本数
    sample_size = 1000  # 每个样本的大小
    random_seed = 420  # 随机种子，用于确保结果可复现
    
    # 设置numpy随机种子
    np.random.seed(random_seed)
    
    # 定义余弦距离计算函数
    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 简化版KMedoids实现
    def simple_kmedoids(vectors, k, seed=random_seed):
        # 随机选择初始中心点
        n = len(vectors)
        rng = np.random.default_rng(seed)
        medoid_indices = rng.choice(n, k, replace=False)
        medoids = vectors[medoid_indices]
        
        # 迭代优化
        max_iter = 100
        for _ in range(max_iter):
            # 分配每个点到最近的中心点
            labels = np.zeros(n, dtype=int)
            for i in range(n):
                distances = [cosine_distance(vectors[i], m) for m in medoids]
                labels[i] = np.argmin(distances)
            
            # 更新中心点
            new_medoid_indices = np.zeros(k, dtype=int)
            for i in range(k):
                cluster_points = vectors[labels == i]
                if len(cluster_points) == 0:
                    continue
                
                # 计算簇内每个点作为中心点的成本
                min_cost = float('inf')
                best_medoid = 0
                
                for j, point in enumerate(cluster_points):
                    cost = sum([cosine_distance(point, p) for p in cluster_points])
                    if cost < min_cost:
                        min_cost = cost
                        best_medoid = j
                
                new_medoid_indices[i] = np.where((vectors == cluster_points[best_medoid]).all(axis=1))[0][0]
            
            # 检查是否收敛
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
            
            medoid_indices = new_medoid_indices
            medoids = vectors[medoid_indices]
        
        # 计算最终成本
        cost = 0
        for i in range(n):
            distances = [cosine_distance(vectors[i], m) for m in medoids]
            cost += min(distances)
        
        return medoid_indices, labels, cost
    
    print(f"正在运行简化版 CLARA 算法: {samples} 个样本，每个样本大小 {sample_size}...")
    
    best_cost = float('inf')
    best_medoids = None
    best_labels = None
    
    for i in tqdm(range(samples), desc="CLARA Samples"):
        # 1. 随机抽取样本
        sample_indices = np.random.choice(len(doc_vectors_norm), sample_size, replace=False)
        sample_vectors = doc_vectors_norm[sample_indices]
        
        # 2. 在样本上运行简化版KMedoids，传递不同的种子以增加多样性
        medoid_indices_in_sample, _, cost = simple_kmedoids(sample_vectors, num_clusters, seed=random_seed + i)
        
        # 3. 保存最佳结果
        if cost < best_cost:
            best_cost = cost
            
            # 获取样本中的中心点在原始数据中的索引
            best_medoids = sample_indices[medoid_indices_in_sample]
    
    # 4. 使用最佳中心点对所有数据进行聚类
    print("正在使用最佳中心点对所有数据进行聚类...")
    clara_labels = np.zeros(len(doc_vectors_norm), dtype=int)
    
    for i in tqdm(range(len(doc_vectors_norm)), desc="Assigning labels"):
        distances = [cosine_distance(doc_vectors_norm[i], doc_vectors_norm[m]) for m in best_medoids]
        clara_labels[i] = np.argmin(distances)
    
    # 统计聚类结果
    cluster_counts = Counter(clara_labels)
    print(f"CLARA 聚类结果统计:")
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"  簇 {cluster_id}: {count} 个样本")
    
    total_clusters = len(set(clara_labels))
    print(f"总聚类数: {total_clusters} 个")

    # ==========================================
    # 6. 降维可视化
    # ==========================================
    print(f"正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto', n_jobs=NUM_CORES)
    vectors_2d = tsne.fit_transform(doc_vectors_norm)

    plot_df = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'CLARA聚类标签': clara_labels,
        'Raw_Text': unique_texts,
        'Processed_Text': corpus_as_strings  # 保存经过词形还原后的文本
    })

    # ==========================================
    # 7. 提取关键词与绘图
    # ==========================================
    print("\n★ 正在分析聚类关键词...")
    cluster_keywords = {}
    stopwords_list = list(stopwords_set) if stopwords_set else []

    # 获取所有唯一的聚类标签
    unique_clusters = sorted(set(clara_labels))
    for c in unique_clusters:
        # 使用经过词形还原后的文本提取关键词
        texts = plot_df[plot_df['CLARA聚类标签'] == c]['Processed_Text'].tolist()
        keywords = get_cluster_keywords(texts, stop_words_list=stopwords_list)
        cluster_keywords[c] = keywords
        print(f"  簇 {c}: [{keywords}]")

    print("\n正在绘图...")
    plt.figure(figsize=(16, 12)) 
    
    # 绘制散点图
    sns.scatterplot(data=plot_df, x='x', y='y', hue='CLARA聚类标签', palette='viridis', s=40, alpha=0.6)
    
    # 添加聚类中心标签和关键词
    for c in unique_clusters:
        cluster_points = plot_df[plot_df['CLARA聚类标签'] == c]
        if len(cluster_points) == 0: continue
        label_text = f"簇{c}\n{cluster_keywords.get(c, '')}"
        plt.text(cluster_points['x'].mean(), cluster_points['y'].mean(), label_text, 
                 ha='center', va='center', fontsize=11, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.title(f'Word2Vec + CLARA 聚类可视化', fontsize=18)
    plt.savefig(os.path.join(save_dir, 'word2vec_lemmatized_clara_figure.png'), dpi=300)
    
    # 将聚类结果保存为CSV文件
    csv_output_path = os.path.join(save_dir, 'clustering_results.csv')
    # 选择需要保存的列：聚类标签和原始文本
    output_df = plot_df[['CLARA聚类标签', 'Raw_Text']]
    output_df.to_csv(csv_output_path, index=False, encoding='utf-8')
    print(f"聚类结果已保存到: {csv_output_path}")
    
    print(f"结果已保存。总耗时: {time() - start_total:.2f} 秒")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()