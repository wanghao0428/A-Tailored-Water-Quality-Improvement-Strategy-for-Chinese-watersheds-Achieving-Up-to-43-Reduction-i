import joblib
import pandas as pd
import numpy as np

# 读取特征值文件
features_df = pd.read_csv('data/train_fenhe.tsv', sep='\t')

# 将特征值转换为模型输入所需的格式
features = features_df['value'].values.reshape(1, -1)

# 加载模型并进行预测
def load_and_predict(model_path, features):
    model = joblib.load(model_path)
    prediction = model.predict(features)
    return prediction

# 模型文件路径
model_paths = ['train_fenhe/r1/gtb.mod', 'train_fenhe/r2/gtb.mod', 'train_fenhe/r3/gtb.mod', 'train_fenhe/r4/gtb.mod']

# 进行预测并生成结果
predictions = []
for path in model_paths:
    pred = load_and_predict(path, features)
    predictions.append(pred)

# 输出预测结果
for i, pred in enumerate(predictions):
    print(f'r{i+1} prediction: {pred}')
