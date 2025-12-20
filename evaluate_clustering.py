import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
import multiprocessing
from collections import Counter

# 从原文件导入需要的函数和模块
from word2vec_kmeans_clustering import (
    preprocess_wrapper,
    get_average_vector,
    load_stopwords,
    NUM_CORES
)

# 导入机器学习库
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 并行处理库
from joblib import Parallel, delayed
from tqdm import tqdm

def evaluate_clustering_k_values():
    """
    评估不同K值(5~12)的聚类效果，输出轮廓系数、CH指数、DBI指数
    """
    start_total = time()
    
    print("====== 开始聚类评估 ======")
    
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

    # 设定是否开启词形还原
    USE_LEMMATIZATION = True 
    
    # 并行处理：预处理文本
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
    # 2. Word2Vec 模型加载
    # ==========================================
    model_path = os.path.join("models", "word2vec_kmeans.model")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行原程序训练模型")
        return
    
    print(f"\n★ 正在加载 Word2Vec 模型: {model_path}")
    model = Word2Vec.load(model_path)
    print("模型加载成功！")

    # ==========================================
    # 3. 生成文档向量 (直接计算词向量平均值)
    # ==========================================
    print("\n★ 正在生成 Word2Vec 平均文档向量...")
    
    vector_size = model.vector_size
    doc_vectors = []
    
    for doc in tqdm(valid_docs, desc="Vectorizing"):
        vec = get_average_vector(doc, model, vector_size)
        doc_vectors.append(vec)
        
    doc_vectors = np.array(doc_vectors)
    doc_vectors_norm = normalize(doc_vectors)

    # ==========================================
    # 4. 评估不同 K 值的聚类效果
    # ==========================================
    print(f"\n====== 开始评估 K 值 5~12 的聚类效果 ======")
    
    # 定义要评估的 K 值范围
    k_values = range(5, 13)
    
    # 存储评估结果
    evaluation_results = []
    
    for k in k_values:
        print(f"\n正在评估 K = {k}...")
        start_k = time()
        
        # 运行 KMeans 聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(doc_vectors_norm)
        
        # 计算评估指标
        silhouette_avg = silhouette_score(doc_vectors_norm, kmeans_labels)
        ch_score = calinski_harabasz_score(doc_vectors_norm, kmeans_labels)
        dbi_score = davies_bouldin_score(doc_vectors_norm, kmeans_labels)
        
        # 计算聚类耗时
        k_time = time() - start_k
        
        # 保存结果
        evaluation_results.append({
            'K值': k,
            '轮廓系数 (Silhouette Score)': silhouette_avg,
            'CH指数 (Calinski-Harabasz Score)': ch_score,
            'DBI指数 (Davies-Bouldin Index)': dbi_score,
            '耗时 (秒)': k_time
        })
        
        # 打印当前 K 值的结果
        print(f"  K = {k} 评估结果:")
        print(f"    轮廓系数: {silhouette_avg:.6f}")
        print(f"    CH指数: {ch_score:.6f}")
        print(f"    DBI指数: {dbi_score:.6f}")
        print(f"    耗时: {k_time:.2f} 秒")
    
    # ==========================================
    # 5. 输出结果
    # ==========================================
    print(f"\n====== 聚类评估结果汇总 ======")
    
    # 创建结果数据框
    results_df = pd.DataFrame(evaluation_results)
    
    # 打印结果表格
    print("\n" + "="*80)
    print("不同 K 值的聚类评估结果")
    print("="*80)
    print(results_df.to_string(index=False, float_format="%.6f"))
    
    # 保存结果到 CSV 文件
    results_csv_path = "clustering_evaluation_results.csv"
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {results_csv_path}")
    
    # 分析最佳 K 值
    print(f"\n====== 最佳 K 值分析 ======")
    
    # 轮廓系数越高越好
    best_silhouette = results_df.loc[results_df['轮廓系数 (Silhouette Score)'].idxmax()]
    print(f"最佳轮廓系数 (越高越好): K = {best_silhouette['K值']}, 得分 = {best_silhouette['轮廓系数 (Silhouette Score)']:.6f}")
    
    # CH 指数越高越好
    best_ch = results_df.loc[results_df['CH指数 (Calinski-Harabasz Score)'].idxmax()]
    print(f"最佳 CH 指数 (越高越好): K = {best_ch['K值']}, 得分 = {best_ch['CH指数 (Calinski-Harabasz Score)']:.6f}")
    
    # DBI 指数越低越好
    best_dbi = results_df.loc[results_df['DBI指数 (Davies-Bouldin Index)'].idxmin()]
    print(f"最佳 DBI 指数 (越低越好): K = {best_dbi['K值']}, 得分 = {best_dbi['DBI指数 (Davies-Bouldin Index)']:.6f}")
    
    total_time = time() - start_total
    print(f"\n总耗时: {total_time:.2f} 秒")
    
    # ==========================================
    # 6. 绘制轮廓系数折线图
    # ==========================================
    print(f"\n====== 绘制轮廓系数折线图 ======")
    
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['K值'], results_df['轮廓系数 (Silhouette Score)'], 'bo-', linewidth=2, markersize=8)
    plt.title('不同K值的轮廓系数变化', fontsize=16)
    plt.xlabel('聚类数 K', fontsize=14)
    plt.ylabel('轮廓系数 (Silhouette Score)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(results_df['K值'])
    
    # 在每个点上标注数值
    for i, row in results_df.iterrows():
        plt.text(row['K值'], row['轮廓系数 (Silhouette Score)'] + 0.005, 
                 f"{row['轮廓系数 (Silhouette Score)']:.4f}", 
                 ha='center', fontsize=10, weight='bold')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('silhouette_score_trend.png', dpi=300, bbox_inches='tight')
    print(f"轮廓系数折线图已保存到: silhouette_score_trend.png")
    
    print(f"====== 聚类评估完成 ======")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    evaluate_clustering_k_values()
