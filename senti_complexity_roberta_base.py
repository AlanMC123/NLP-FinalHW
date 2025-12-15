import os
# 屏蔽底层日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
import time
from scipy.special import softmax
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 0. 全局配置
# ==========================================
SAVE_DIR = 'senti_complexity_roberta'
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# ★关键设置：Batch Size★
BATCH_SIZE = 64 

# 绘图风格
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
except:
    pass

# 创建根目录
os.makedirs(SAVE_DIR, exist_ok=True)

PARTY_COLORS = {
    'Conservative': '#0087DC',                   
    'Labour': '#E4003B',                         
    'Labourco-Operative': '#E4003B',             
    'Green': '#6AB023',                          
    'Scottish-National-Party': '#9B59B6',
    'Liberal-Democrat': '#FF9F43',
    'Plaid-Cymru': '#16A085',
    'Dup': '#A93226',
    'Ukip': '#D2B4DE',
    'Social-Democratic-And-Labour-Party': '#2ECC71',
    'Uup': '#48C9B0',
    'Alliance': '#F1C40F',
    'Respect': '#34495E',
    'Independent': '#95A5A6',
    'Independent-Conservative': '#5D6D7E',
    'Independent-Ulster-Unionist': '#85929E'
}

# ==========================================
# 1. GPU 推理核心模块 (Batch Processing)
# ==========================================
def run_gpu_inference(df):
    """
    使用 GPU 进行批量推理
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> 硬件检测: 正在使用 {device} 进行推理")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    
    texts = df['speech'].tolist()
    total_len = len(texts)
    sentiment_scores = []
    
    print(">>> 启动 GPU 加速推理...")
    
    for i in tqdm(range(0, total_len, BATCH_SIZE), desc="GPU Inference", unit="batch"):
        batch_texts = texts[i : i + BATCH_SIZE]
        try:
            encoded_input = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                output = model(**encoded_input)
            
            scores = output.logits.detach().cpu().numpy()
            probs = softmax(scores, axis=1)
            # col 0: Neg, col 2: Pos -> Score = Pos - Neg
            batch_sentiments = probs[:, 2] - probs[:, 0]
            sentiment_scores.extend(batch_sentiments)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n!!! 显存溢出 (OOM) 在索引 {i}。请调小 BATCH_SIZE。")
                torch.cuda.empty_cache()
                exit()
            else:
                print(f"Error: {e}")
                sentiment_scores.extend([0.0] * len(batch_texts))

    return sentiment_scores

# ==========================================
# 2. 绘图通用函数
# ==========================================
def generate_plots(df, metric_col, output_subfolder, metric_name="Sentiment"):
    """
    Args:
        df: 数据集
        metric_col: 要分析的列名 ('sentiment' 或 'sentiment_zscore')
        output_subfolder: 输出子文件夹名 ('raw' 或 'z-score')
        metric_name: 图表中显示的指标名称
    """
    # 确保子目录存在
    save_path_root = os.path.join(SAVE_DIR, output_subfolder)
    os.makedirs(save_path_root, exist_ok=True)
    
    print(f"    正在生成图表到: {save_path_root} ...")
    
    palette_dict = {p: PARTY_COLORS.get(p, '#808080') for p in df['party'].unique()}
    
    # 聚合每个发言人的数据
    speaker_df_all = df.groupby(['speaker_name', 'party'])[[metric_col, 'complexity']].mean().reset_index()

    # 计算全局轴范围 (用于统一单个政党图的坐标轴)
    global_limit = max(abs(speaker_df_all[metric_col].min()), abs(speaker_df_all[metric_col].max())) * 1.1
    if pd.isna(global_limit) or global_limit == 0: global_limit = 1.0

    # ---------------------------------------------------------
    # 1. 拆分 Speaker 图片，按政党输出
    # ---------------------------------------------------------
    unique_parties = df['party'].unique()
    for party_name in unique_parties:
        sub_df = speaker_df_all[speaker_df_all['party'] == party_name]
        if len(sub_df) == 0: continue

        plt.figure(figsize=(12, 8))
        color = PARTY_COLORS.get(party_name, '#808080')
        
        sns.scatterplot(data=sub_df, x=metric_col, y='complexity', color=color, s=120, alpha=0.8)
        plt.axvline(0, color='red', linestyle='--')
        
        plt.title(f'Speaker Style: {party_name}', fontsize=18)
        plt.xlabel(f'{metric_name} Polarity', fontsize=14)
        plt.ylabel('Complexity (Grade Level)', fontsize=14)
        
        # 统一坐标轴范围
        plt.xlim(-global_limit, global_limit)
        
        plt.tight_layout()
        safe_name = "".join([c if c.isalnum() else "_" for c in party_name])
        plt.savefig(os.path.join(save_path_root, f'speaker_{safe_name}.png'), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # 2. Party 气泡图 (聚合视图)
    # ---------------------------------------------------------
    party_df = df.groupby('party').agg({metric_col: 'mean', 'complexity': 'mean', 'speech': 'count'}).reset_index()
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=party_df, x=metric_col, y='complexity', hue='party', size='speech', sizes=(500, 5000), alpha=0.75, palette=palette_dict, legend=False)
    
    for _, row in party_df.iterrows():
        plt.text(row[metric_col], row['complexity'], row['party'], ha='center', va='center', weight='bold')
    
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'Party Analysis ({metric_name})', fontsize=18)
    plt.xlabel(f'{metric_name} Polarity', fontsize=14)
    
    # 限制横轴范围 (根据原始代码要求)
    plt.xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_root, 'party_bubble.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. [新增] 全体发言人散点图 (All Speakers) - 带图例
    # ---------------------------------------------------------
    plt.figure(figsize=(16, 10)) # 宽度增加以容纳图例
    
    # 使用所有发言人数据
    sns.scatterplot(
        data=speaker_df_all, 
        x=metric_col, 
        y='complexity', 
        hue='party',         # 颜色区分政党
        palette=palette_dict,
        s=60,                # 点稍微小一点以免过于拥挤
        alpha=0.6,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.title(f'All Speakers: {metric_name} vs Complexity', fontsize=20)
    plt.xlabel(f'{metric_name} Polarity', fontsize=15)
    plt.ylabel('Complexity (Grade Level)', fontsize=15)
    
    # 放置图例在右侧外
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., title='Party')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_root, 'all_speakers_scatter.png'), dpi=300)
    plt.close()

# ==========================================
# 3. 主程序
# ==========================================
def main():
    start_time = time.time()
    
    # 读取数据
    csv_path = 'corpus/ParlVote_concat.csv'
    print(f"\n>>> 读取数据: {csv_path}")
    try:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, sep='\t')
                if 'speech' not in df.columns: df = pd.read_csv(csv_path)
            except: df = pd.read_csv(csv_path)
        else: raise FileNotFoundError
    except:
        print("⚠️ 使用模拟数据...")
        parties = ['Labour', 'Conservative', 'SNP', 'Liberal Democrat', 'Green']
        data = [{'speech': "This is a test speech." * 5, 'speaker_name': f'MP_{i%20}', 'party': parties[i%5]} for i in range(2000)]
        df = pd.DataFrame(data)

    df = df.dropna(subset=['speech', 'party']).copy()
    df['speech'] = df['speech'].astype(str)
    df['party'] = df['party'].apply(lambda x: str(x).title())
    
    print(f"数据总量: {len(df)} 行")

    # --- 步骤 1: GPU 情感分析 (得到原始分数) ---
    # 原始分数范围通常在 -1 (Neg) 到 1 (Pos) 之间
    df['sentiment'] = run_gpu_inference(df)
    
    # --- 步骤 2: 文本复杂度 (CPU计算) ---
    print("\n>>> 计算文本复杂度 (CPU)...")
    tqdm.pandas(desc="Complexity")
    df['complexity'] = df['speech'].progress_apply(lambda x: textstat.flesch_kincaid_grade(str(x)) if len(str(x).split()) > 3 else 0.0)

    # --- 步骤 3: Z-Score 标准化 ---
    mean_val = df['sentiment'].mean()
    std_val = df['sentiment'].std()
    df['sentiment_zscore'] = (df['sentiment'] - mean_val) / std_val
    
    # 保存结果 CSV
    out_path = os.path.join(SAVE_DIR, 'roberta_results.csv')
    df.to_csv(out_path, index=False)
    
    # --- 步骤 4: 绘图 (修改部分) ---
    if len(df) > 0:
        print("\n>>> 开始生成图表...")

        # 1. 输出原始分数图片 (Raw)
        generate_plots(
            df=df, 
            metric_col='sentiment', 
            output_subfolder='raw', 
            metric_name='Raw Sentiment Score'
        )

        # 2. 输出 Z-Score 图片 (Z-Score)
        generate_plots(
            df=df, 
            metric_col='sentiment_zscore', 
            output_subfolder='z-score', 
            metric_name='Sentiment Z-Score'
        )

    print(f"\n★ 全部完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"★ 结果位置: {os.path.abspath(SAVE_DIR)}")

if __name__ == '__main__':
    main()