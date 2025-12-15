from gensim.models import Word2Vec
import os

# 加载Word2Vec模型
def load_word2vec_model(model_path):
    if os.path.exists(model_path):
        print(f"正在加载Word2Vec模型: {model_path}")
        try:
            model = Word2Vec.load(model_path)
            print("模型加载成功！")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    else:
        print(f"模型文件不存在: {model_path}")
        return None

# 查询相似单词
def get_similar_words(model, word, topn=20):
    if model is None:
        return []
    
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        return similar_words
    except KeyError:
        print(f"单词 '{word}' 不在词表中！")
        return []

# 主函数
def main():
    # 模型路径
    model_path = 'clustering_analysis/word2vec.model'
    
    # 加载模型
    model = load_word2vec_model(model_path)
    
    if model is not None:
        # 获取用户输入
        while True:
            word = input("请输入一个单词 (输入 'exit' 退出): ").strip().lower()
            if word == 'exit':
                print("程序已退出。")
                break
            
            # 查询相似单词
            similar_words = get_similar_words(model, word)
            
            if similar_words:
                print(f"\n与 '{word}' 最相似的 {len(similar_words)} 个单词：")
                for i, (similar_word, similarity) in enumerate(similar_words, 1):
                    print(f"{i}. {similar_word}: 相似度 = {similarity:.4f}")
            print()

if __name__ == '__main__':
    main()