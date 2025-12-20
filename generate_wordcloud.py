import os
import re
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 创建保存文件夹
output_dir = 'wordcloud'
os.makedirs(output_dir, exist_ok=True)

# 读取文件内容
file_path = 'bertopic_analysis_None/bertopic_party_topics.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 定义正则表达式匹配模式
party_pattern = r'政党 (.+?):'
# 匹配主题行：提取比例和关键词
# 示例：  1. 主题 7: 比例=95.65%, 关键词=ireland, northern, decommission, ira, agreement
# 匹配分组：1. 比例值  2. 关键词列表
# 注意：要匹配多行，所以使用 re.DOTALL
# 每个主题行的模式

topic_pattern = r'\s+\d+\. 主题 \d+: 比例=(\d+\.\d+)%, 关键词=(.+?)\n'

# 提取所有政党
parties = re.findall(party_pattern, content)

# 格式化政党名称：一般政党名字单词首字母大写，缩写全部大写
def format_party_name(party_name):
    # 常见的政党缩写列表
    abbreviations = ['DUP', 'UUP', 'SNP', 'UKIP']
    
    # 检查是否在缩写列表中
    if party_name.upper() in abbreviations:
        return party_name.upper()
    
    # 检查是否是缩写（包含连字符，且每个部分都是1-2个字母）
    parts = party_name.split('-')
    is_abbreviation = all(len(part) <= 2 for part in parts)
    if is_abbreviation:
        return party_name.upper()
    
    # 检查是否已经是全部大写
    if party_name.isupper():
        return party_name
    
    # 一般情况：单词首字母大写
    # 处理连字符分隔的情况
    if '-' in party_name:
        formatted_parts = []
        for part in parts:
            formatted_parts.append(part.capitalize())
        return '-'.join(formatted_parts)
    
    # 处理空格分隔的情况
    elif ' ' in party_name:
        formatted_words = []
        for word in party_name.split(' '):
            formatted_words.append(word.capitalize())
        return ' '.join(formatted_words)
    
    # 单个单词的情况
    else:
        return party_name.capitalize()

# 提取每个政党的主题数据
def extract_party_topics(party_name, content):
    # 找到该政党的起始位置
    party_start = content.find(f'政党 {party_name}:')
    if party_start == -1:
        return []
    
    # 找到下一个政党的起始位置（如果有的话），或者文件结尾
    next_party = content.find('政党 ', party_start + len(f'政党 {party_name}:'))
    if next_party == -1:
        party_content = content[party_start:]
    else:
        party_content = content[party_start:next_party]
    
    # 提取该政党的所有主题
    topics = re.findall(topic_pattern, party_content)
    return topics

# 生成词云图
def generate_wordcloud(party_name, topics):
    # 格式化政党名称
    formatted_party_name = format_party_name(party_name)
    
    # 构建词频字典
    word_freq = {}
    
    for proportion_str, keywords_str in topics:
        proportion = float(proportion_str)
        # 跳过比例为0的主题
        if proportion == 0:
            continue
        
        # 计算权重：y = ln(x+1)，其中x是比例值（已经是百分比，无需转换为小数）
        weight = math.log(proportion + 1)
        
        # 提取关键词
        keywords = keywords_str.split(', ')
        
        # 为每个关键词添加权重
        for word in keywords:
            # 跳过空关键词
            if not word:
                continue
            if word in word_freq:
                word_freq[word] += weight
            else:
                word_freq[word] = weight
    
    # 如果没有关键词，跳过
    if not word_freq:
        return False
    
    # 创建词云对象
    wc = WordCloud(
        width=800,
        height=600,
        background_color='white',
        max_words=100,
        collocations=False,
        font_path='C:\\Windows\\Fonts\\msyh.ttc',  # 使用微软雅黑字体
        margin=20  # 增加四周留白，默认值是2
    )
    
    # 生成词云
    wc.generate_from_frequencies(word_freq)
    
    # 绘制词云图
    plt.figure(figsize=(8, 6))
    
    # 设置matplotlib的默认字体为微软雅黑
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'sans-serif']
    
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{formatted_party_name} 词云图', fontsize=16)
    
    # 保存词云图
    output_path = os.path.join(output_dir, f'{formatted_party_name}_wordcloud.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 恢复默认字体设置
    plt.rcParams['font.family'] = ['sans-serif']
    
    print(f'✅ 已生成 {formatted_party_name} 词云图，保存到 {output_path}')
    return True

# 主程序
if __name__ == '__main__':
    print('开始生成词云图...')
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个政党
    for party in parties:
        topics = extract_party_topics(party, content)
        generate_wordcloud(party, topics)
    
    print(f'\n词云图生成完成！所有词云图已保存到 {output_dir} 文件夹')
