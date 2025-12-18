import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import nltk
import re
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings('ignore')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception:
    pass

lemmatizer = WordNetLemmatizer()

# ==========================================
# 1. 核心数配置
# ==========================================
TOTAL_CORES = multiprocessing.cpu_count()
if TOTAL_CORES > 4:
    NUM_CORES = TOTAL_CORES - 4
else:
    NUM_CORES = max(1, TOTAL_CORES - 1)
NUM_CORES = 10
CLUSTER_NUM = None   # 指定聚类数量，None 表示自动聚类

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  

result_folder = f'bertopic_analysis_{CLUSTER_NUM}'
os.makedirs(result_folder, exist_ok=True)

# ==========================================
# 辅助函数
# ==========================================
# 使用 bert_stopwords.txt 作为主要停用词表
stopwords_path = 'stopwords/bert_stopwords.txt'
all_stopwords = []
if os.path.exists(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        all_stopwords = [line.strip() for line in f if line.strip()]
    all_stopwords = set(all_stopwords)
else:
    # 如果 bert_stopwords.txt 不存在，使用默认的 nltk 停用词
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    all_stopwords = set(nltk_stopwords)

def preprocess_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    # 定义需要保留原始大小写的政党名称
    party_names = ['Conservative', 'Labour', 'Liberal-Democrat', 'Scottish-National-Party', 'Plaid-Cymru', 
                  'Labourco-operative', 'UUP', 'DUP', 'Green', 'Independent', 
                  'Social-Democratic-and-Labour-Party', 'Respect', 'UKIP', 'Alliance', 
                  'Independent-Conservative', 'Independent-Ulster-Unionist']
    
    # 将文本转换为字符串
    text = str(text)
    
    # 移除所有标点符号和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 转换为小写
    text = text.lower()
    
    # 分词并处理每个单词
    words = []
    for word in text.split():
        # 检查是否为政党名称（不区分大小写）
        is_party = False
        for party in party_names:
            if word == party.lower():
                # 保留原始大小写
                words.append(party)
                is_party = True
                break
        
        # 如果不是政党名称
        if not is_party:
            # 词形还原 (动词还原，例如 voted -> vote)
            w = lemmatizer.lemmatize(word, pos='v')
            # 再次还原 (名词还原，例如 parties -> party)
            w = lemmatizer.lemmatize(w, pos='n')
            
            words.append(w)
    
    # 严格过滤停用词和短词
    filtered_words = []
    for word in words:
        # 过滤停用词
        if word in all_stopwords:
            continue
        # 过滤短词
        if len(word) < 3:
            continue
        # 过滤数字词
        if word.isdigit():
            continue
        filtered_words.append(word)
    
    # 返回字符串而不是列表，因为 BERTopic 需要原始文本或句子列表
    return ' '.join(filtered_words)

def plot_word_frequency(data, party_col, processed_text_col, prefix, result_folder):
    if party_col not in data.columns: return
    valid_data = data[data[party_col].notna()]
    if valid_data.empty: return

    major_parties = valid_data[party_col].value_counts().head(5).index.tolist()
    
    def get_top_words(party, n=20):
        party_data = valid_data[valid_data[party_col] == party]
        party_texts = party_data[processed_text_col]
        if party_texts.empty: return pd.Series(dtype='int64')
        all_words = [word for text in party_texts for word in text.split()]
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

# ==========================================
# 核心训练函数
# ==========================================
def train_bertopic_and_analyze(data, text_col, party_col, year_col, prefix, result_folder, num_topics=8):
    print(f"\n{'='*50}")
    print(f"正在对 {prefix} 语料库训练 BERTopic 模型...")
    
    texts = data[text_col].tolist()
    print(f"语料库大小: {len(texts)} 文档")
    
    # 初始化 BERTopic 模型，调整参数以提高主题质量
    # 取消指定聚类数，让模型自行决定最优主题数量（nr_topics=None）
    model = BERTopic(
        nr_topics=num_topics,       # 自动决定主题数量
        language='english',
        calculate_probabilities=True,
        verbose=True,
        min_topic_size=100,  # 增加最小主题大小，过滤小主题
        top_n_words=10,       # 只显示前10个关键词
        n_gram_range=(1, 1),  # 不考虑词组
        low_memory=True       # 内存优化
    )
    
    # 训练模型
    topics, probabilities = model.fit_transform(texts)
    
    # 将主题分配添加到数据中
    data[f'{prefix}_topic'] = topics
    data[f'{prefix}_topic_probability'] = [max(prob) if len(prob) > 0 else 0 for prob in probabilities]
    
    # 保存主题信息
    topic_info = model.get_topic_info()
    print(f"主题信息:\n{topic_info}")
    
    # 保存主题描述
    topic_file_path = os.path.join(result_folder, f'{prefix}_topic_results.txt')
    with open(topic_file_path, 'w', encoding='utf-8') as f:
        f.write("BERTopic 主题结果\n")
        f.write("=" * 30 + "\n\n")
        for idx in topic_info.Topic:
            if idx != -1:  # 排除异常主题
                f.write(f"主题 {idx}:\n")
                topic_terms = model.get_topic(idx)
                for term, weight in topic_terms:
                    f.write(f"  {term}: {weight:.4f}\n")
                f.write("\n")
    
    # 保存年度和政党Top5话题的文件
    top_topics_file = os.path.join(result_folder, f'{prefix}_top_topics.txt')
    
    # ==========================================
    # 可视化
    # ==========================================
    print(f">>> 开始 {party_col} 分布分析 <<<")
    
    analysis_data = data[
        data[party_col].notna() & 
        (data[party_col] != '') & 
        (data[f'{prefix}_topic'] != -1) 
    ].copy()
    
    print(f"参与绘图的有效数据行数: {len(analysis_data)}")
    
    if not analysis_data.empty:
        try:
            party_topic_dist = analysis_data.groupby(party_col)[f'{prefix}_topic'].value_counts(normalize=True).unstack(fill_value=0)
            
            plt.figure(figsize=(12, 8))
            party_topic_dist.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
            plt.title(f'各党派主题分布')
            plt.xlabel('党派')
            plt.ylabel('比例')
            plt.legend(title='主题', loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.xticks(rotation=45, ha='right')
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
            
            # 输出每个政党的Top5话题
            print("\n>>> 生成每个政党Top5话题 <<<")
            party_topics_file = os.path.join(result_folder, f'{prefix}_party_topics.txt')
            with open(party_topics_file, 'w', encoding='utf-8') as f:
                f.write("BERTopic 政党Top5话题分析\n")
                f.write("=" * 50 + "\n\n")
                
                # 每个政党的Top5话题
                f.write("1. 每个政党Top5话题\n")
                f.write("-" * 30 + "\n\n")
                
                # 遍历每个政党
                for party in sorted(party_topic_dist.index):
                    # 获取该政党的主题分布，按比例降序排列
                    party_topics = party_topic_dist.loc[party].sort_values(ascending=False)
                    top_5_topics = party_topics.head(5)
                    
                    f.write(f"政党 {party}:\n")
                    for i, (topic_id, proportion) in enumerate(top_5_topics.items(), 1):
                        # 获取主题关键词
                        topic_terms = model.get_topic(topic_id)
                        keywords = [term for term, _ in topic_terms[:5]]
                        f.write(f"  {i}. 主题 {topic_id}: 比例={proportion:.2%}, 关键词={', '.join(keywords)}\n")
                    f.write("\n")
            print(f"✅ 政党Top5话题已保存到: {party_topics_file}")
            
        except Exception as e:
            print(f"❌ 绘图失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 年份主题分析
    print(f"\n>>> 开始 {year_col} 分布分析 <<<")
    
    year_analysis_data = data[
        data[year_col].notna() & 
        (data[f'{prefix}_topic'] != -1) 
    ].copy()
    
    print(f"参与年份分析的有效数据行数: {len(year_analysis_data)}")
    
    if not year_analysis_data.empty:
        try:
            # 计算每年的主题分布
            year_topic_counts = year_analysis_data.groupby(year_col)[f'{prefix}_topic'].value_counts().unstack(fill_value=0)
            year_topic_dist = year_analysis_data.groupby(year_col)[f'{prefix}_topic'].value_counts(normalize=True).unstack(fill_value=0)
            
            # 找出每年的最大主题
            annual_max_topics = year_topic_counts.idxmax(axis=1)
            annual_max_proportions = year_topic_counts.max(axis=1) / year_topic_counts.sum(axis=1)
            
            # 保存每年最大主题到文件
            year_max_topic_file = os.path.join(result_folder, f'{prefix}_annual_max_topics.txt')
            with open(year_max_topic_file, 'w', encoding='utf-8') as f:
                f.write("年度最大主题分析\n")
                f.write("=" * 30 + "\n")
                f.write(f"{'年份':<8} {'最大主题':<10} {'主题比例':<10} {'总文档数':<10}\n")
                f.write("-" * 40 + "\n")
                
                for year in sorted(annual_max_topics.index):
                    max_topic = annual_max_topics[year]
                    proportion = annual_max_proportions[year]
                    total_docs = year_topic_counts.loc[year].sum()
                    f.write(f"{year:<8} {max_topic:<10} {proportion:.2%}{'':<10} {total_docs:<10}\n")
            
            print("✅ 年度最大主题分析已保存")
            
            # 可视化年份-主题分布
            plt.figure(figsize=(14, 8))
            year_topic_dist.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
            plt.title(f'每年主题分布')
            plt.xlabel('年份')
            plt.ylabel('比例')
            plt.legend(title='主题', loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{result_folder}/{prefix}_year_topic_distribution.png')
            plt.close()
            print("✅ 年份-主题分布柱状图已保存")
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(year_topic_dist, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title(f'主题-年份分布热力图')
            plt.xlabel('主题')
            plt.ylabel('年份')
            plt.tight_layout()
            plt.savefig(f'{result_folder}/{prefix}_topic_year_heatmap.png')
            plt.close()
            print("✅ 主题-年份分布热力图已保存")
            
            # 可视化每年最大主题
            plt.figure(figsize=(12, 6))
            years = sorted(annual_max_topics.index)
            max_topics = [annual_max_topics[year] for year in years]
            max_props = [annual_max_proportions[year] for year in years]
            
            plt.bar(years, max_props, color='skyblue')
            for i, (year, topic) in enumerate(zip(years, max_topics)):
                plt.text(year, max_props[i] + 0.01, f'T{topic}', ha='center', fontsize=9)
            
            plt.title('每年最大主题比例')
            plt.xlabel('年份')
            plt.ylabel('最大主题比例')
            plt.ylim(0, 1)
            plt.xticks(years, rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'{result_folder}/{prefix}_annual_max_topic_trend.png')
            plt.close()
            print("✅ 年度最大主题趋势图已保存")
            
            # 输出每年的Top5话题
            print("\n>>> 生成每年Top5话题 <<<")
            with open(top_topics_file, 'w', encoding='utf-8') as f:
                f.write("BERTopic 年度与政党Top5话题分析\n")
                f.write("=" * 50 + "\n\n")
                
                # 1. 每年的Top5话题
                f.write("1. 每年Top5话题\n")
                f.write("-" * 30 + "\n\n")
                
                # 遍历每一年
                for year in sorted(year_topic_counts.index):
                    # 获取该年的主题分布，按数量降序排列
                    year_topics = year_topic_counts.loc[year].sort_values(ascending=False)
                    top_5_topics = year_topics.head(5)
                    
                    f.write(f"年份 {year}:\n")
                    for i, (topic_id, count) in enumerate(top_5_topics.items(), 1):
                        proportion = year_topic_dist.loc[year, topic_id]
                        # 获取主题关键词
                        topic_terms = model.get_topic(topic_id)
                        keywords = [term for term, _ in topic_terms[:5]]
                        f.write(f"  {i}. 主题 {topic_id}: 文档数={count}, 比例={proportion:.2%}, 关键词={', '.join(keywords)}\n")
                    f.write("\n")
            print("✅ 年度Top5话题已保存")
        except Exception as e:
            print(f"❌ 年份分析失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有足够的数据用于年份分析。")

    # BERTopic 特有的可视化
    print("\n>>> 生成 BERTopic 特有可视化 <<<")
    
    # 主题词云
    try:
        fig = model.visualize_topics()
        fig.write_html(os.path.join(result_folder, f'{prefix}_topic_visualization.html'))
        print("✅ 主题可视化 HTML 已保存")
    except Exception as e:
        print(f"❌ 主题可视化失败: {e}")
    
    # 主题层次树
    try:
        fig = model.visualize_hierarchy()
        fig.write_html(os.path.join(result_folder, f'{prefix}_topic_hierarchy.html'))
        print("✅ 主题层次树 HTML 已保存")
    except Exception as e:
        print(f"❌ 主题层次树失败: {e}")
    
    # 主题相似度热力图
    try:
        fig = model.visualize_heatmap()
        fig.write_html(os.path.join(result_folder, f'{prefix}_topic_heatmap.html'))
        print("✅ 主题相似度热力图 HTML 已保存")
    except Exception as e:
        print(f"❌ 主题相似度热力图失败: {e}")
    
    # ==========================================
    # 计算模型评估指标
    # ==========================================
    print("\n>>> 计算模型评估指标 <<<")
    
    # 1. 主题一致性 (Coherence Score)
    def calculate_bertopic_coherence(model, texts, top_n=10, coherence_method='c_v', workers=NUM_CORES):
        """为 BERTopic 模型计算主题一致性，全量数据多线程版本"""
        # 获取主题词列表
        topics = model.get_topics()
        topic_words = []
        for topic_id, words in topics.items():
            if topic_id != -1:  # 排除异常主题
                topic_words.append([word for word, _ in words[:top_n]])
        
        # 为主题一致性计算准备语料库
        tokenized_texts = [text.split() for text in texts]
        
        # 创建字典
        dictionary = corpora.Dictionary(tokenized_texts)
        
        # 根据方法选择不同的计算方式
        if coherence_method == 'u_mass':
            # u_mass 方法需要语料库
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            coherence_model = CoherenceModel(topics=topic_words, corpus=corpus, dictionary=dictionary, coherence='u_mass', processes=workers)
        else:
            # c_v 或其他方法，使用多线程加速
            coherence_model = CoherenceModel(topics=topic_words, texts=tokenized_texts, dictionary=dictionary, coherence=coherence_method, processes=workers)
        
        return coherence_model.get_coherence()
    
    # 使用全量数据和多线程计算主题一致性
    coherence_score = calculate_bertopic_coherence(model, texts, coherence_method='c_v', workers=NUM_CORES)
    print(f"🔍 主题一致性 (Coherence Score, c_v): {coherence_score:.4f}")
    
    # 2. 主题多样性 (Topic Diversity)
    def calculate_topic_diversity(model, top_n=10):
        """计算主题多样性：不同主题中唯一词的比例"""
        topics = model.get_topics()
        all_words = set()
        total_words = 0
        
        for topic_id, words in topics.items():
            if topic_id != -1:  # 排除异常主题
                topic_words = [word for word, _ in words[:top_n]]
                all_words.update(topic_words)
                total_words += len(topic_words)
        
        if total_words == 0:
            return 0.0
        
        return len(all_words) / total_words
    
    topic_diversity = calculate_topic_diversity(model)
    print(f"🔍 主题多样性 (Topic Diversity): {topic_diversity:.4f}")
    
    # 3. 困惑度 (Perplexity) - BERTopic 版本，全量数据版本
    def calculate_bertopic_perplexity(probabilities):
        """基于概率分布计算 BERTopic 模型的困惑度，使用全量数据"""
        # 过滤有效概率
        valid_probs = [prob for prob in probabilities if isinstance(prob, np.ndarray) and len(prob) > 0]
        
        if not valid_probs:
            return 0.0
        
        # 使用全量数据，向量化计算熵，提高效率
        valid_probs_array = np.array(valid_probs)
        entropy = -np.sum(valid_probs_array * np.log(valid_probs_array + 1e-12), axis=1)
        
        # 困惑度是平均熵的指数
        avg_entropy = np.mean(entropy)
        perplexity = np.exp(avg_entropy)
        return perplexity
    
    perplexity = calculate_bertopic_perplexity(probabilities)
    print(f"🔍 困惑度 (Perplexity): {perplexity:.4f}")
    
    # 保存指标到文件
    metrics_file = os.path.join(result_folder, f'{prefix}_model_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("BERTopic 模型评估指标\n")
        f.write("=" * 30 + "\n")
        f.write(f"主题一致性 (Coherence Score): {coherence_score:.4f}\n")
        f.write(f"主题多样性 (Topic Diversity): {topic_diversity:.4f}\n")
        f.write(f"困惑度 (Perplexity): {perplexity:.4f}\n")
    print(f"✅ 模型指标已保存到: {metrics_file}")
    
    # 保存模型
    model_dir = 'models/'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'bertopic_model'))
    print(f"✅ BERTopic 模型已保存到: {model_dir}/bertopic_model")
    
    return data, model

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
        df['combined_text'] = df['processed_motion'] + ' ' + df['processed_speech']
        
        # 从 debate_id 提取年份
        df['year'] = df['debate_id'].astype(str).str[:4].astype(int)
        print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
        
        # 过滤
        df_filtered = df[df['combined_text'].apply(len) > 0].copy()
        print(f"过滤后数据量: {len(df_filtered)}")

        # 训练
        train_bertopic_and_analyze(
            df_filtered, 
            text_col='combined_text', 
            party_col='motion_party', 
            year_col='year',  
            prefix='bertopic',  
            result_folder=result_folder,
            num_topics=CLUSTER_NUM,
        )
        
        print("\n程序完成！")
    else:
        print(f"找不到文件 {csv_path}")