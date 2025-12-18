import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# 设置设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载情感分析模型和分词器
# 使用预训练的情感分析模型
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# 设置模型为推理模式
model.eval()

def calculate_sentiment_score(text):
    """
    计算文本的情感分数：正面概率 - 负面概率
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # 模型输出：[负面概率, 中性概率, 正面概率]
    negative_prob = probabilities[0]
    neutral_prob = probabilities[1]
    positive_prob = probabilities[2]
    # 情感分数 = 正面概率 - 负面概率
    sentiment_score = positive_prob - negative_prob
    return sentiment_score

def main():
    # 1. 读取聚类结果CSV文件
    clustering_csv_path = "clustering_analysis_10/clustering_results.csv"
    print(f"正在读取聚类结果: {clustering_csv_path} ...")
    
    if not os.path.exists(clustering_csv_path):
        print(f"错误: 文件 {clustering_csv_path} 不存在！")
        print("请先运行 word2vec_clustering.py 生成聚类结果。")
        return
    
    df = pd.read_csv(clustering_csv_path)
    print(f"读取到 {len(df)} 条数据")
    
    # 2. 计算每个文本的情感分数
    print("\n正在计算情感分数...")
    sentiment_scores = []
    
    # 使用tqdm显示进度
    for text in tqdm(df['Raw_Text'], desc="情感分析"):
        score = calculate_sentiment_score(text)
        sentiment_scores.append(score)
    
    # 将情感分数添加到DataFrame
    df['Sentiment_Score'] = sentiment_scores
    
    # 3. 计算每个簇的平均情感水平与方差
    print("\n正在计算每个簇的情感统计数据...")
    cluster_stats = df.groupby('KMeans聚类标签')['Sentiment_Score'].agg(['mean', 'std']).reset_index()
    cluster_stats.columns = ['Cluster', 'Average_Sentiment', 'Sentiment_Variance']
    
    # 4. 输出结果
    print("\n====== 各簇情感统计结果 ======")
    print(cluster_stats.to_string(index=False))
    
    # 5. 保存结果到CSV
    output_csv_path = "clustering_analysis_10/cluster_sentiment_stats.csv"
    cluster_stats.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n情感统计结果已保存到: {output_csv_path}")
    
    # 也保存包含情感分数的完整数据
    full_output_path = "clustering_analysis_10/clustering_with_sentiment.csv"
    df.to_csv(full_output_path, index=False, encoding='utf-8')
    print(f"包含情感分数的完整数据已保存到: {full_output_path}")

if __name__ == "__main__":
    main()