import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
import re
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# ==========================================
# 1. 核心数配置
# ==========================================
TOTAL_CORES = multiprocessing.cpu_count()
if TOTAL_CORES > 4:
    NUM_CORES = TOTAL_CORES - 4
else:
    NUM_CORES = max(1, TOTAL_CORES - 1)

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  

result_folder = 'lda_analysis'
os.makedirs(result_folder, exist_ok=True)

try:
    nltk.download('punkt', quiet=True)
except Exception:
    pass

# ==========================================
# 辅助函数
# ==========================================
stopwords_path = 'stopwords.txt'
custom_stopwords = []
if os.path.exists(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        custom_stopwords = [line.strip() for line in f if line.strip()]

nltk_stopwords = stopwords.words('english')
all_stopwords = set(nltk_stopwords + custom_stopwords)

def preprocess_text(text):
    if pd.isna(text) or text is None:
        return []
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = simple_preprocess(text, deacc=True)
    # 稍微放宽长度限制，保留2个字母的词
    words = [word for word in words if word not in all_stopwords and len(word) >= 2]
    return words

def plot_word_frequency(data, party_col, processed_text_col, prefix, result_folder):
    if party_col not in data.columns: return
    valid_data = data[data[party_col].notna()]
    if valid_data.empty: return

    major_parties = valid_data[party_col].value_counts().head(5).index.tolist()
    
    def get_top_words(party, n=20):
        party_data = valid_data[valid_data[party_col] == party]
        party_texts = party_data[processed_text_col]
        if party_texts.empty: return pd.Series(dtype='int64')
        all_words = [word for text in party_texts for word in text]
        return pd.Series(all_words).value_counts().head(n)
    
    try:
        plt.figure(figsize=(20, 15))
        if len(major_parties) > 0:
            rows = (len(major_parties) + 1) // 2
            for i, party in enumerate(major_parties, 1):
                plt.subplot(rows, 2, i)
                word_freq = get_top_words(party, 15)
                if len(word_freq) > 0:
                    sns.barplot(x=word_freq.values, y=word_freq.index.astype(str), hue=word_freq.index.astype(str), palette='viridis', legend=False)
                    plt.title(f'{party} 高频词')
                    plt.xlabel('词频')
                    plt.yticks(fontsize=10)
            plt.tight_layout(pad=2.0)
            plt.savefig(f'{result_folder}/{prefix}_party_word_frequency.png', dpi=300)
            plt.close()
        
        # 保存文本
        all_parties = valid_data[party_col].unique().tolist()
        word_freq_file = os.path.join(result_folder, f'{prefix}_all_party_word_freq.txt')
        with open(word_freq_file, 'w', encoding='utf-8') as f:
            for party in all_parties:
                word_freq = get_top_words(party, 20)
                f.write(f"\n{party} 词频统计:\n")
                for word, freq in word_freq.items():
                    f.write(f"{word}: {freq}\n")
    except Exception as e:
        print(f"词频绘图错误: {e}")

def extract_dominant_topic_safe(topic_dist):
    """从主题分布列表中提取概率最大的主题ID"""
    if not topic_dist or not isinstance(topic_dist, list):
        return -1
    try:
        best_topic = max(topic_dist, key=lambda x: x[1])
        return best_topic[0]
    except:
        return -1

# ==========================================
# 核心训练函数 (已修复)
# ==========================================
def train_combined_lda_and_analyze(data, combined_text_col, party_col, prefix, result_folder, num_topics=6):
    print(f"\n{'='*50}")
    print(f"正在对 {prefix} 语料库训练 LDA 模型...")
    
    texts = data[combined_text_col].tolist()
    dictionary = corpora.Dictionary(texts)
    
    print(f"原始词汇表大小: {len(dictionary)}")
    
    # --- 修改 1: 极度放宽过滤条件，或者直接注释掉 ---
    # 之前可能 no_below=5 删除了太多词，现在改为 1 或者 2
    # no_above=0.9 表示除非 90% 的文档都有这个词，否则保留
    dictionary.filter_extremes(no_below=2, no_above=0.9) 
    print(f"过滤后词汇表大小: {len(dictionary)}")
    
    if len(dictionary) == 0:
        print("❌ 错误：词汇表为空！请检查数据预处理步骤。")
        return data, None, None, None

    # 构建语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # 检查语料库是否全是空的 (Debug)
    empty_docs = sum(1 for doc in corpus if not doc)
    print(f"空文档向量数量: {empty_docs} / {len(corpus)}")
    if empty_docs == len(corpus):
        print("❌ 错误：所有文档的词袋向量都为空！这意味着预处理后的词都不在词典里。")
        return data, None, None, None

    print(f"正在训练 LDA 模型 (主题数: {num_topics}, 核心数: {NUM_CORES})...")
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        chunksize=2000,
        passes=5,
        alpha='symmetric',
        per_word_topics=True,
        workers=NUM_CORES
    )
    
    # 保存主题
    topic_file_path = os.path.join(result_folder, f'{prefix}_topic_results.txt')
    with open(topic_file_path, 'w', encoding='utf-8') as f:
        for idx, topic in lda_model.print_topics(-1):
            f.write(f"主题 {idx}: {topic}\n")

    # ==========================================
    # 修复重点：计算分布 (强制输出概率)
    # ==========================================
    print(f"\n正在计算文档主题分布...")
    
    # --- 修改 2: 使用 get_document_topics 并设置 minimum_probability=0 ---
    # 这样即使概率很低，也不会返回空列表
    topic_distributions = []
    # 这里不能用多进程，因为 lda_model 传递开销大，直接单线程跑循环，速度通常可以接受
    for i in tqdm(range(len(corpus)), desc="推断主题", ncols=80, unit="doc"):
        # minimum_probability=0 强制返回完整分布
        dist = lda_model.get_document_topics(corpus[i], minimum_probability=0)
        topic_distributions.append(dist)
    
    data[f'{prefix}_topic_distribution'] = pd.Series(topic_distributions, index=data.index)
    
    print("正在提取主要主题...")
    data[f'{prefix}_dominant_topic'] = data[f'{prefix}_topic_distribution'].map(extract_dominant_topic_safe)
    
    # 再次检查
    valid_topics_count = data[f'{prefix}_dominant_topic'].value_counts()
    print(f"主题分布统计 (检查点): \n{valid_topics_count.head()}")
    
    if -1 in valid_topics_count.index and valid_topics_count[-1] == len(data):
        print("❌ 依然未能提取到有效主题。请检查数据质量。")
        return data, lda_model, dictionary, corpus

    # ==========================================
    # 可视化
    # ==========================================
    print(f"\n>>> 开始 {party_col} 分布分析 <<<")
    
    analysis_data = data[
        data[party_col].notna() & 
        (data[party_col] != '') & 
        (data[f'{prefix}_dominant_topic'] != -1) 
    ].copy()
    
    print(f"参与绘图的有效数据行数: {len(analysis_data)}")
    
    if not analysis_data.empty:
        try:
            party_topic_dist = analysis_data.groupby(party_col)[f'{prefix}_dominant_topic'].value_counts(normalize=True).unstack(fill_value=0)
            
            plt.figure(figsize=(12, 8))
            party_topic_dist.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
            plt.title(f'各党派主题分布')
            plt.xlabel('党派')
            plt.ylabel('比例')
            plt.legend(title='主题', loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{result_folder}/{prefix}_party_topic_distribution.png')
            plt.close()
            print("✅ 柱状图已保存")
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(party_topic_dist, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title(f'主题-党派分布热力图')
            plt.xlabel('主题')
            plt.ylabel('党派')
            plt.tight_layout()
            plt.savefig(f'{result_folder}/{prefix}_topic_party_heatmap.png')
            plt.close()
            print("✅ 热力图已保存")
            
        except Exception as e:
            print(f"❌ 绘图失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有足够的数据用于绘图。")

    print("\n正在进行词频统计...")
    plot_word_frequency(data, party_col, combined_text_col, prefix, result_folder)
    
    return data, lda_model, dictionary, corpus

# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    print(f"物理核心: {TOTAL_CORES}, 使用核心: {NUM_CORES}")
    csv_path = 'corpus/ParlVote_concat.csv'
    
    if os.path.exists(csv_path):
        print("正在读取数据...")
        df = pd.read_csv(csv_path, usecols=['debate_id', 'motion_party', 'debate_title', 'motion_text', 'party', 'speech'])
        
        # 预处理
        print("预处理 Motion...")
        motion_texts = df['motion_text'].tolist()
        df['processed_motion'] = Parallel(n_jobs=NUM_CORES)(delayed(preprocess_text)(t) for t in tqdm(motion_texts, ncols=80, unit="doc"))
        
        print("预处理 Speech...")
        speech_texts = df['speech'].tolist()
        df['processed_speech'] = Parallel(n_jobs=NUM_CORES)(delayed(preprocess_text)(t) for t in tqdm(speech_texts, ncols=80, unit="doc"))
        
        # 合并
        print("合并语料...")
        combined = [m + s for m, s in zip(df['processed_motion'], df['processed_speech'])]
        df['combined_text'] = combined
        
        # 过滤
        df_combined = df[df['combined_text'].apply(len) > 0].copy()
        print(f"过滤后数据量: {len(df_combined)}")

        # 训练
        train_combined_lda_and_analyze(
            df_combined, 
            combined_text_col='combined_text', 
            party_col='motion_party', 
            prefix='lda',  
            result_folder=result_folder,
            num_topics=7
        )
        
        print("\n程序完成！")
    else:
        print(f"找不到文件 {csv_path}")