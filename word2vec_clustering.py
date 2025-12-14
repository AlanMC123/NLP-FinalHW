import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import os
from time import time
from collections import Counter  # ★ 新增：用于计算TF

# NLP & 机器学习库
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer 

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

def load_stopwords(filepath='stopwords/w2v_stopwords.txt'):
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

def preprocess_wrapper(text, stopwords_set=None):
    if pd.isna(text):
        return []
    tokens = simple_preprocess(str(text))
    if stopwords_set:
        tokens = [t for t in tokens if t not in stopwords_set]
    return tokens

def get_cluster_keywords(texts, stop_words_list=None, top_n=5):
    """提取聚类关键词 (TF-IDF)"""
    if not texts: return "N/A"
    base_stop_words = 'english'
    final_stop_words = base_stop_words
    if stop_words_list and len(stop_words_list) > 0:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        final_stop_words = list(ENGLISH_STOP_WORDS.union(stop_words_list))

    tfidf = TfidfVectorizer(stop_words=final_stop_words, max_features=1000)
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = np.array(tfidf.get_feature_names_out())
        avg_weights = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = avg_weights.argsort()[::-1][:top_n]
        return ", ".join(feature_names[top_indices])
    except ValueError:
        return "N/A"

# ★ 修改：计算 TF-IDF 加权平均文档向量
def get_tfidf_weighted_vector(doc_tokens, w2v_model, idf_dict, vector_size):
    """
    获取文档向量：计算 TF-IDF 加权平均值
    Formula: sum(Vec_w * TF_w * IDF_w) / sum(TF_w * IDF_w)
    """
    # 筛选出既在 Word2Vec 模型中，又在 IDF 字典中的词
    valid_tokens = [word for word in doc_tokens if word in w2v_model.wv.key_to_index and word in idf_dict]
    
    if not valid_tokens:
        return np.zeros(vector_size)
    
    # 计算当前文档的 TF (词频)
    tf_counter = Counter(valid_tokens)
    total_tokens = len(valid_tokens)
    
    weighted_sum = np.zeros(vector_size)
    total_weight = 0.0
    
    for word, count in tf_counter.items():
        # 获取词向量
        vec = w2v_model.wv[word]
        
        # 计算权重: TF * IDF
        # TF = count / total_tokens (分母在归一化时其实会被抵消，直接用 count * idf 也可以，但为了严谨写全)
        tf = count / total_tokens
        idf = idf_dict[word]
        weight = tf * idf
        
        weighted_sum += vec * weight
        total_weight += weight
        
    if total_weight == 0:
        return np.zeros(vector_size)
        
    return weighted_sum / total_weight

def find_optimal_k_elbow(doc_vectors, max_k=15, save_dir='clustering_analysis'):
    """肘部法自动寻找最佳K"""
    print(f"\n====== 正在运行肘部法 (Elbow Method) ======")
    wcss = []
    K_range = range(1, max_k + 1)
    
    for k in tqdm(K_range, desc="Calculating WCSS"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(doc_vectors)
        wcss.append(kmeans.inertia_)
    
    # 几何距离法寻找拐点
    x = np.array(K_range)
    y = np.array(wcss)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    start_point = np.array([x_norm[0], y_norm[0]])
    end_point = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end_point - start_point
    
    distances = []
    for i in range(len(x)):
        point = np.array([x_norm[i], y_norm[i]])
        dist = np.abs(np.cross(line_vec, point - start_point)) / np.linalg.norm(line_vec)
        distances.append(dist)
    
    best_k = K_range[np.argmax(distances)]
    print(f"★ 自动检测到的最佳聚类数 (Elbow Point): K = {best_k}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss, 'bo-', label='WCSS')
    plt.plot(best_k, wcss[np.argmax(distances)], 'ro', markersize=12, label=f'Optimal K={best_k}')
    plt.title('The Elbow Method for Optimal k (Word2Vec + TFIDF)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'elbow_method_curve_tfidf.png'), dpi=300)
    
    return best_k

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

    # 加载停用词
    stopwords_set = load_stopwords('stopwords/w2v_stopwords.txt')

    # ==========================================
    # 2. 并行分词
    # ==========================================
    print(f"正在预处理文本 (分词/去停用词)...")
    tokenized_docs = Parallel(n_jobs=NUM_CORES)(
        delayed(preprocess_wrapper)(text, stopwords_set) for text in tqdm(unique_texts, desc="Tokenizing")
    )
    
    # 过滤空文档
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
    save_dir = 'clustering_analysis'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'word2vec.model')
    
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
    # ★ 4. 计算 TF-IDF 并生成加权向量
    # ==========================================
    print("\n★ 正在准备 TF-IDF 权重...")
    
    # TfidfVectorizer 需要字符串列表，而不是 token 列表，所以这里先 join 回去
    corpus_as_strings = [" ".join(doc) for doc in valid_docs]
    
    # 初始化并拟合 TF-IDF
    # 使用 min_df 过滤掉极罕见词，防止噪声干扰权重
    tfidf_vectorizer = TfidfVectorizer(min_df=2) 
    tfidf_vectorizer.fit(corpus_as_strings)
    
    # 创建 {word: idf_value} 字典，以便快速查找
    # 注意：get_feature_names_out() 返回的是排序后的词汇表
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_values = tfidf_vectorizer.idf_
    idf_dict = dict(zip(feature_names, idf_values))
    
    print(f"TF-IDF 词表大小: {len(idf_dict)}")
    print("正在计算 TF-IDF 加权文档向量...")
    
    # 这里的 vector_size 必须与 Word2Vec 模型一致
    vector_size = model.vector_size
    
    doc_vectors = []
    # 使用 tqdm 显示进度，因为加权计算比简单平均稍微慢一点点
    for doc in tqdm(valid_docs, desc="Vectorizing"):
        vec = get_tfidf_weighted_vector(doc, model, idf_dict, vector_size)
        doc_vectors.append(vec)
        
    doc_vectors = np.array(doc_vectors)
    
    # 归一化 (对聚类至关重要)
    doc_vectors_norm = normalize(doc_vectors)

    # ==========================================
    # 5. 肘部法与聚类
    # ==========================================
    # best_k = find_optimal_k_elbow(doc_vectors_norm, max_k=15, save_dir=save_dir)
    best_k = 6

    print(f"\n正在使用最佳 K={best_k} 运行最终聚类...")
    kmeans = KMeans(n_clusters=best_k, random_state=2026, n_init=10)
    kmeans_labels = kmeans.fit_predict(doc_vectors_norm)

    # ==========================================
    # 6. 降维可视化
    # ==========================================
    print(f"正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto', n_jobs=NUM_CORES)
    vectors_2d = tsne.fit_transform(doc_vectors_norm)

    plot_df = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'KMeans_Labels': kmeans_labels,
        'Raw_Text': unique_texts
    })

    # ==========================================
    # 7. 提取关键词与绘图
    # ==========================================
    print("\n★ 正在分析聚类关键词...")
    cluster_keywords = {}
    stopwords_list = list(stopwords_set) if stopwords_set else []

    for c in range(best_k):
        texts = plot_df[plot_df['KMeans_Labels'] == c]['Raw_Text'].tolist()
        keywords = get_cluster_keywords(texts, stop_words_list=stopwords_list)
        cluster_keywords[c] = keywords
        print(f"  Cluster {c}: [{keywords}]")

    print("\n正在绘图...")
    plt.figure(figsize=(16, 12)) 
    sns.scatterplot(data=plot_df, x='x', y='y', hue='KMeans_Labels', palette='viridis', s=40, alpha=0.6)
    
    for c in range(best_k):
        cluster_points = plot_df[plot_df['KMeans_Labels'] == c]
        if len(cluster_points) == 0: continue
        label_text = f"C{c}\n{cluster_keywords.get(c, '')}"
        plt.text(cluster_points['x'].mean(), cluster_points['y'].mean(), label_text, 
                 ha='center', va='center', fontsize=11, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.title(f'Word2Vec (TF-IDF Weighted) Clustering (Optimal K={best_k})', fontsize=18)
    plt.savefig(os.path.join(save_dir, 'word2vec_tfidf_kmeans_figure.png'), dpi=300)
    print(f"结果已保存。总耗时: {time() - start_total:.2f} 秒")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()