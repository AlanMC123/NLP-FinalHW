import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import os
import re  # 新增正则库
from time import time
from collections import Counter

# NLP & 机器学习库
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer

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

NUM_CORES = max(1, multiprocessing.cpu_count() - 2)

# ==========================================
# 1. 新增：强力清洗函数
# ==========================================

import re

def clean_parliamentary_text(text, is_motion=False):
    """
    [强力版] 清洗议会文本
    1. 移除称谓 (Mr Speaker)
    2. 移除程序性交互 (Give way)
    3. 移除套话 (I beg to move)
    4. 标记纯噪音 (Rubbish)
    """
    if pd.isna(text):
        return ""
    
    # 1. 基础清理
    text = str(text).strip()
    # 移除括号内的内容 (通常是动作描述，如 [Interruption] 或 [Laughter])
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    text_lower = text.lower()
    
    # ==========================
    # A. 针对 Motion (动议) 的清洗
    # ==========================
    if is_motion:
        # 动议常见的“无意义”起手式
        motion_patterns = [
            r"^(?:i\s+)?beg\s+to\s+move,?\s*(?:that)?", 
            r"^that\s+this\s+house\s+(?:notes|believes|regrets|deplores),?",
            r"^that\s+the\s+clause\s+be\s+read\s+a\s+second\s+time,?",
            r"^clause\s+\d+.*?", # 以 Clause X 开头的通常是引用
            r"^page\s+\d+,\s+line\s+\d+.*?", # 页码引用
            r"amendment\s+no\.\s+\d+"
        ]
        for pat in motion_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # ==========================
    # B. 针对 Speech (发言) 的清洗
    # ==========================
    else:
        # 1. 移除议会特有的称谓 (这些词频率极高但无实质语义)
        addressing_patterns = [
            r"mr\s+speaker,?", 
            r"madam\s+deputy\s+speaker,?", 
            r"mr\.\s+speaker,?",
            r"hon\.\s+gentleman", 
            r"hon\.\s+lady",
            r"right\s+hon\.\s+member",
            r"my\s+hon\.\s+friend"
        ]
        for pat in addressing_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)

        # 2. 移除程序性打断/交互 (Interventions)
        intervention_patterns = [
            r"will\s+the\s+(?:.*)\s+give\s+way\??", # Will the gentleman give way?
            r"i\s+give\s+way",
            r"on\s+a\s+point\s+of\s+order",
            r"thank\s+you,?\s+mr\s+speaker"
        ]
        for pat in intervention_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)

        # 3. 检查是否为“纯噪音”短语
        # 如果剩下的话只是这些，直接清空
        noise_phrases = {
            "hear", "hear hear", "hear, hear", "agreed", 
            "rubbish", "shame", "disgrace", "resign", 
            "order", "sit down", "withdraw", "no", "aye"
        }
        
        # 简单的分词检查，如果清洗完只剩噪音词，则返回空
        cleaned_words = [w for w in re.split(r'\W+', text_lower) if w]
        if not cleaned_words: 
            return ""
        if len(cleaned_words) <= 3 and all(w in noise_phrases for w in cleaned_words):
            return ""

    # 最后的修剪：去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 2. 原有辅助函数
# ==========================================

def load_stopwords(filepath='stopwords/w2v_stopwords.txt'):
    """只从文件读取停用词"""
    stopwords = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f if line.strip())
            print(f"成功加载停用词表，共 {len(stopwords)} 个词。")
        except Exception as e:
            print(f"加载停用词表失败: {e}")
    else:
        print(f"警告: 未找到停用词文件 {filepath}")
    return stopwords

def preprocess_wrapper(text, stopwords_set=None):
    """
    预处理：分词 -> 去停用词 -> 词干化
    """
    if pd.isna(text) or text == "":
        return []
    
    # 1. 基础分词
    tokens = simple_preprocess(str(text))
    
    # 2. 去停用词 (如果传入了集合)
    if stopwords_set:
        tokens = [t for t in tokens if t not in stopwords_set]
    
    # 3. 词干化
    stemmer = SnowballStemmer("english")
    stemmed_tokens = [stemmer.stem(t) for t in tokens]
    
    return stemmed_tokens

def get_tfidf_weighted_vector(doc_tokens, w2v_model, idf_dict, vector_size):
    """计算 TF-IDF 加权平均向量"""
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

# ==========================================
# Main 函数
# ==========================================

def main():
    start_total = time()
    
    save_dir = 'consistency_analysis'
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据读取
    csv_path = 'corpus/ParlVote_concat.csv'
    print(f"正在读取数据: {csv_path} ...")
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 初步去空
            df = df.dropna(subset=['motion_text', 'speech']).reset_index(drop=True)
            print(f"原始有效数据量: {len(df)}")
        else:
            raise FileNotFoundError("CSV file not found")
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # ---------------------------------------------------------
    # ★★★ 新增步骤：强力数据清洗 ★★★
    # ---------------------------------------------------------
    print("\n[清洗阶段] 正在去除议会套话与非实质性发言...")
    
    # A. 应用文本清洗函数
    # 注意：先清洗文本，去除 "I beg to move" 等前缀
    df['motion_clean'] = df['motion_text'].apply(lambda x: clean_parliamentary_text(x, is_motion=True))
    df['speech_clean'] = df['speech'].apply(lambda x: clean_parliamentary_text(x, is_motion=False))
    
    # B. 计算单词数量 (简单估算)
    df['motion_len'] = df['motion_clean'].apply(lambda x: len(str(x).split()))
    df['speech_len'] = df['speech_clean'].apply(lambda x: len(str(x).split()))
    
    # C. 强力过滤条件
    # Motion > 3: 保证动议有实质内容 (不仅是 "I move clause 6")
    # Speech > 8: 保证发言是论述性的 (过滤掉 "Rubbish!", "Give way" 等短语)
    mask_valid = (df['motion_len'] > 3) & (df['speech_len'] > 8)
    
    # 查看被剔除的样本 (Debug用)
    dropped_count = len(df) - mask_valid.sum()
    if dropped_count > 0:
        print(f"\n剔除低质量数据: {dropped_count} 条 (占比 {dropped_count/len(df):.1%})")
        print("--- 被剔除样本示例 ---")
        print(df[~mask_valid][['motion_text', 'speech']].head(3))
        print("----------------------\n")
    
    # 执行剔除
    df = df[mask_valid].reset_index(drop=True)
    print(f"清洗后剩余数据量: {len(df)}")

    # ---------------------------------------------------------
    # 停用词设置：传入空集合 (不使用停用词，依靠TF-IDF降权)
    # ---------------------------------------------------------
    stopwords_set = set() 
    # 如果你想恢复使用停用词，取消下面注释：
    # stopwords_set = load_stopwords('stopwords/w2v_stopwords.txt')

    # 2. 并行分词 (使用清洗后的列 motion_clean, speech_clean)
    print("正在处理文本 (分词 -> 词干化)...")
    
    df['motion_tokens'] = Parallel(n_jobs=NUM_CORES)(
        delayed(preprocess_wrapper)(t, stopwords_set) for t in tqdm(df['motion_clean'], desc="Processing Motion")
    )
    df['speech_tokens'] = Parallel(n_jobs=NUM_CORES)(
        delayed(preprocess_wrapper)(t, stopwords_set) for t in tqdm(df['speech_clean'], desc="Processing Speech")
    )

    all_sentences = df['motion_tokens'].tolist() + df['speech_tokens'].tolist()

    # 3. Word2Vec 模型处理
    possible_models = [os.path.join('w2v_models', 'word2vec_con.model')]
    model = None
    target_model_path = 'w2v_models/word2vec_con.model'

    for path in possible_models:
        if os.path.exists(path):
            print(f"\n★ 检测到已有模型: {path}")
            try:
                model = Word2Vec.load(path)
                print(">>> 模型加载成功！")
                break
            except Exception as e:
                print(f">>> 加载失败 ({e})")
    
    if model is None:
        print(f"\n未找到可用模型，正在训练新 Word2Vec 模型 (语料句数: {len(all_sentences)})...")
        model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=2, workers=NUM_CORES, seed=42, sg=1)
        model.save(target_model_path)
        print(f"模型已保存至: {target_model_path}")

    # 4. 计算 TF-IDF
    print("\n正在计算 TF-IDF...")
    corpus_as_strings = [" ".join(tokens) for tokens in all_sentences]
    
    tfidf_vectorizer = TfidfVectorizer(min_df=2)
    tfidf_vectorizer.fit(corpus_as_strings)
    
    idf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    vector_size = model.vector_size

    # 5. 计算一致性 (Cosine Similarity)
    print("\n★ 正在计算 Motion 与 Speech 的语义一致性...")
    
    similarity_scores = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating"):
        vec_motion = get_tfidf_weighted_vector(row['motion_tokens'], model, idf_dict, vector_size)
        vec_speech = get_tfidf_weighted_vector(row['speech_tokens'], model, idf_dict, vector_size)
        
        if np.all(vec_motion == 0) or np.all(vec_speech == 0):
            sim = 0.0 
        else:
            sim = cosine_similarity(vec_motion.reshape(1, -1), vec_speech.reshape(1, -1))[0][0]
            
        similarity_scores.append(sim)

    df['consistency_score'] = similarity_scores

    # 6. 保存与绘图
    output_csv = os.path.join(save_dir, 'consistency_results.csv') # 修改文件名以示区别
    cols_to_save = ['motion_text', 'speech', 'consistency_score']
    for col in ['party', 'debate_id', 'vote', 'date', 'year']:
        if col in df.columns:
            cols_to_save.append(col)
            
    df[cols_to_save].to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")

    # 绘制分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['consistency_score'], bins=30, kde=True, color='skyblue')
    plt.title('动议-发言一致性得分分布（清洗后数据）')
    plt.xlabel('余弦相似度')
    plt.ylabel('数量')
    plt.axvline(df['consistency_score'].mean(), color='r', linestyle='--', label=f'平均值: {df["consistency_score"].mean():.3f}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'consistency_distribution.png'), dpi=300)
    
    # 绘制政党箱线图
    if 'party' in df.columns:
        top_parties = df['party'].value_counts().head(10).index
        plot_data = df[df['party'].isin(top_parties)]
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='party', y='consistency_score', data=plot_data, palette='viridis')
        plt.title('各政党一致性得分对比（清洗后）')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'consistency_by_party.png'), dpi=300)

    print("\n====== 分析完成 ======")
    print(f"平均一致性: {df['consistency_score'].mean():.4f}")
    print(f"总耗时: {time() - start_total:.2f} 秒")
    
    # 依然查看最低分，确认是否还有垃圾数据
    low_consistency_df = df.sort_values(by='consistency_score').head(10)
    print("=== 相关性最低的 10 个样本 (清洗后) ===")
    for idx, row in low_consistency_df.iterrows():
        print(f"Party: {row['party']}")
        print(f"Score: {row['consistency_score']:.4f}")
        print(f"Motion (Clean): {row['motion_clean'][:100]}...") 
        print(f"Speech (Clean): {row['speech_clean'][:100]}...") 
        print("-" * 30)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()