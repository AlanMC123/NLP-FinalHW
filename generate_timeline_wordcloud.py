import os
import re
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 创建保存文件夹
output_dir = 'timeline_wordcloud'
os.makedirs(output_dir, exist_ok=True)

# 读取文件内容
file_path = 'bertopic_analysis_None/bertopic_top_topics.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有年份的主题数据
def extract_yearly_topics(content):
    # 修复后的主题提取函数，能正确匹配主题行
    yearly_topics = {}
    
    # 查找所有年份行的位置
    year_matches = re.finditer(r'年份 (\d+):', content)
    years = [int(match.group(1)) for match in year_matches]
    
    for year in years:
        # 查找该年份的起始位置
        year_start = content.find(f'年份 {year}:')
        
        # 查找下一个年份的起始位置
        next_year_start = content.find(f'年份 {year+1}:')
        if next_year_start == -1 and year != years[-1]:
            # 找下一个年份
            next_year_start = content.find(f'年份 {years[years.index(year)+1]}:')
        if next_year_start == -1:
            next_year_start = len(content)
        
        # 提取该年份的内容
        year_content = content[year_start:next_year_start]
        
        # 提取主题行：匹配 "  1. 主题 3: 文档数=43, 比例=16.73%, 关键词=..."
        topic_matches = re.findall(r'\s+\d+\. 主题 \d+: 文档数=\d+, 比例=(\d+\.\d+)%, 关键词=(.+?)\n', year_content)
        yearly_topics[year] = topic_matches
    
    return yearly_topics

# 生成单年词云图
def generate_year_wordcloud(year, topics, output_dir, size=(400, 300)):
    # 构建词频字典
    word_freq = {}
    
    for proportion_str, keywords_str in topics:
        proportion = float(proportion_str)
        # 使用对数缩放（y=lnx），加1避免ln(0)错误，并乘以10放大权重
        weight = math.log(proportion + 1) * 10
        
        # 提取关键词
        keywords = keywords_str.split(', ')
        
        # 为每个关键词添加权重
        for word in keywords:
            if word in word_freq:
                word_freq[word] += weight
            else:
                word_freq[word] = weight
    
    # 如果没有关键词，跳过
    if not word_freq:
        print(f"年份 {year} 没有足够的关键词生成词云图")
        return None
    
    # 检查词频情况
    print(f"年份 {year} 词频统计：")
    print(f"  总词数：{len(word_freq)}")
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, freq in top_words:
        print(f"  {word}: {freq:.2f}")
    
    # 创建词云对象
    wc = WordCloud(
        width=size[0],
        height=size[1],
        background_color='white',
        max_words=100,
        collocations=False,
        font_path=r'C:\Windows\Fonts\msyh.ttc',  # 使用微软雅黑字体支持中文
        margin=20,
        min_font_size=10,
        relative_scaling=1.0,
        scale=2
    )
    
    # 生成词云
    wc.generate_from_frequencies(word_freq)
    
    # 保存单年词云图（调试用）
    single_output_path = os.path.join(output_dir, f'{year}_wordcloud.png')
    wc.to_file(single_output_path)
    print(f"  单年词云已保存到：{single_output_path}")
    
    return wc

# 创建时间轴词云图
def create_timeline_wordcloud(yearly_topics, output_dir):
    # 按年份排序
    sorted_years = sorted(yearly_topics.keys())
    num_years = len(sorted_years)
    
    # 设置时间轴图尺寸
    # 每个词云的尺寸
    cloud_width = 400
    cloud_height = 300
    
    # 总宽度：每个词云宽度 + 间距
    total_width = num_years * cloud_width
    total_height = cloud_height + 100  # 额外空间用于年份标签
    
    # 创建大图
    plt.figure(figsize=(total_width / 100, total_height / 100), dpi=100)
    gs = GridSpec(2, num_years, height_ratios=[cloud_height, 50])
    
    for i, year in enumerate(sorted_years):
        topics = yearly_topics[year]
        wc = generate_year_wordcloud(year, topics, output_dir)
        
        if wc is not None:
            # 绘制词云
            ax_cloud = plt.subplot(gs[0, i])
            ax_cloud.imshow(wc, interpolation='bilinear')
            ax_cloud.axis('off')
        else:
            # 空的子图
            ax_cloud = plt.subplot(gs[0, i])
            ax_cloud.axis('off')
        
        # 绘制年份标签
        ax_year = plt.subplot(gs[1, i])
        ax_year.axis('off')
        ax_year.text(0.5, 0.5, str(year), ha='center', va='center', fontsize=12, fontproperties={'family': 'Microsoft YaHei'})
    
    # 调整布局
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    
    # 保存时间轴图
    timeline_path = os.path.join(output_dir, 'yearly_wordcloud_timeline.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return timeline_path

# 主程序
if __name__ == '__main__':
    print('开始生成时间轴词云图...')
    
    # 解析年度主题数据
    yearly_topics = extract_yearly_topics(content)
    print(f'已解析 {len(yearly_topics)} 个年份的数据')
    
    # 生成时间轴词云图
    timeline_path = create_timeline_wordcloud(yearly_topics, output_dir)
    print(f'✅ 时间轴词云图已保存到: {timeline_path}')
    
    print('程序完成！')