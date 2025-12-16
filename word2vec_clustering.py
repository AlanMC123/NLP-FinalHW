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
from sklearn.cluster import KMeans
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

def find_optimal_k_elbow(doc_vectors, max_k=15, save_dir='clustering_analysis_8'):
    """肘部法自动寻找最佳K"""
    print(f"\n====== 正在运行肘部法 (Elbow Method) ======")
    wcss = []
    K_range = range(1, max_k + 1)
    
    for k in tqdm(K_range, desc="Calculating WCSS"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(doc_vectors)
        wcss.append(kmeans.inertia_)
    
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

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss, 'bo-', label='WCSS (簇内平方和)')
    plt.plot(best_k, wcss[np.argmax(distances)], 'ro', markersize=12, label=f'最佳K值={best_k}')
    plt.title('肘部法确定最佳聚类数 (Word2Vec + TFIDF)')
    plt.xlabel('聚类数 (k)')
    plt.ylabel('WCSS (簇内平方和)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'elbow_method_curve_tfidf.png'), dpi=300)
    
    return best_k

def main(k):
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
    save_dir = f'clustering_analysis_{k}'
    os.makedirs(save_dir, exist_ok=True)
    # 修改模型文件名以区分是否使用了词干化
    model_name = 'word2vec.model'
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
    # 4. 计算 TF-IDF 并生成加权向量
    # ==========================================
    print("\n★ 正在准备 TF-IDF 权重...")
    
    corpus_as_strings = [" ".join(doc) for doc in valid_docs]
    
    tfidf_vectorizer = TfidfVectorizer(min_df=2) 
    tfidf_vectorizer.fit(corpus_as_strings)
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_values = tfidf_vectorizer.idf_
    idf_dict = dict(zip(feature_names, idf_values))
    
    print(f"TF-IDF 词表大小: {len(idf_dict)}")
    print("正在计算 TF-IDF 加权文档向量...")
    
    vector_size = model.vector_size
    doc_vectors = []
    
    for doc in tqdm(valid_docs, desc="Vectorizing"):
        vec = get_tfidf_weighted_vector(doc, model, idf_dict, vector_size)
        doc_vectors.append(vec)
        
    doc_vectors = np.array(doc_vectors)
    doc_vectors_norm = normalize(doc_vectors)

    # ==========================================
    # 5. 肘部法与聚类
    # ==========================================
    best_k = find_optimal_k_elbow(doc_vectors_norm, max_k=15, save_dir=save_dir)
    best_k = k

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
        'KMeans聚类标签': kmeans_labels,
        'Raw_Text': unique_texts,
        'Processed_Text': corpus_as_strings  # 保存经过词形还原后的文本
    })

    # ==========================================
    # 7. 提取关键词与绘图
    # ==========================================
    print("\n★ 正在分析聚类关键词...")
    cluster_keywords = {}
    stopwords_list = list(stopwords_set) if stopwords_set else []

    for c in range(best_k):
        # 使用经过词形还原后的文本提取关键词
        texts = plot_df[plot_df['KMeans聚类标签'] == c]['Processed_Text'].tolist()
        keywords = get_cluster_keywords(texts, stop_words_list=stopwords_list)
        cluster_keywords[c] = keywords
        print(f"  C{c}: [{keywords}]")

    print("\n正在绘图...")
    plt.figure(figsize=(16, 12)) 
    sns.scatterplot(data=plot_df, x='x', y='y', hue='KMeans聚类标签', palette='viridis', s=40, alpha=0.6)
    
    for c in range(best_k):
        cluster_points = plot_df[plot_df['KMeans聚类标签'] == c]
        if len(cluster_points) == 0: continue
        label_text = f"簇{c}\n{cluster_keywords.get(c, '')}"
        plt.text(cluster_points['x'].mean(), cluster_points['y'].mean(), label_text, 
                 ha='center', va='center', fontsize=11, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.title(f'Word2Vec (TF-IDF加权) 聚类可视化', fontsize=18)
    save_filename = 'word2vec_lemmatized_tfidf_figure.png' if USE_LEMMATIZATION else 'word2vec_tfidf_figure.png'
    plt.savefig(os.path.join(save_dir, save_filename), dpi=300)
    print(f"结果已保存。总耗时: {time() - start_total:.2f} 秒")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main(10)