import os
import pickle
import pandas as pd


def test_models():
    models_dir = 'models'

    if not os.path.exists(models_dir):
        print("模型目录不存在，请先运行 train_model.py")
        return False

    # 检查所有必需的文件是否存在
    required_files = [
        'movies_dict.pkl',
        'movies2_dict.pkl',
        'new_df_dict.pkl',
        'similarity_tags_tags.pkl',
        'similarity_tags_genres.pkl',
        'similarity_tags_keywords.pkl',
        'similarity_tags_tcast.pkl',
        'similarity_tags_tprduction_comp.pkl'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(models_dir, file)):
            missing_files.append(file)

    if missing_files:
        print(f"缺少以下模型文件: {missing_files}")
        return False

    print("所有必需模型文件都存在")

    # 测试加载数据
    try:
        # 加载 new_df
        with open(os.path.join(models_dir, 'new_df_dict.pkl'), 'rb') as f:
            new_df_dict = pickle.load(f)
        new_df = pd.DataFrame.from_dict(new_df_dict)
        print(f"成功加载 new_df，包含 {len(new_df)} 部电影")

        # 加载相似度矩阵
        with open(os.path.join(models_dir, 'similarity_tags_tags.pkl'), 'rb') as f:
            similarity_matrix = pickle.load(f)
        print(f"成功加载相似度矩阵，形状: {similarity_matrix.shape}")

        # 显示一些基本信息
        print(f"\n数据基本信息:")
        print(f"- 电影数量: {len(new_df)}")
        print(f"- 特征列: {list(new_df.columns)}")
        print(f"- 前5部电影: {list(new_df['title'].head())}")

        return True

    except Exception as e:
        print(f"加载模型文件时出错: {e}")
        return False


if __name__ == '__main__':
    print("开始测试模型文件...")
    success = test_models()
    if success:
        print("\n模型测试通过！")
    else:
        print("\n模型测试失败！")
