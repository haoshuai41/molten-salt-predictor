import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # 用于保存和加载模型
import os

def load_excel_data(excel_file_path):
    """
    从Excel文件加载数据
    """
    df = pd.read_excel(excel_file_path)
    
    # 定义特征列名
    feature_names = ['NaCl', 'KCl', 'MgCl2', 'temperature']
    
    # 提取特征数据
    X = df[feature_names].values
    
    # 创建目标变量字典
    y = {
        'density': df['density'].values,
        'heat_capacity': df['heat_capacity'].values,
        'viscosity': df['viscosity'].values,
        'thermal_conductivity': df['thermal_conductivity'].values
    }
    
    # 转换为DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"从Excel加载了 {len(df)} 个样本")
    return X_df, y

class MoltenSaltPropertyPredictor:
    """
    NaCl-KCl-MgCl2熔盐热物性参数机器学习预测模型
    """
    def __init__(self):
        """初始化模型和属性名称"""
        self.models = {}
        self.feature_names = ['NaCl', 'KCl', 'MgCl2', 'temperature']
        self.property_names = ['density', 'heat_capacity', 'viscosity', 'thermal_conductivity']
        self.model_info = {}  # 存储模型性能信息

    def load_data(self, excel_file_path):
        """
        从Excel文件加载数据
        """
        return load_excel_data(excel_file_path)

    def train_models(self, X, y):
        """训练随机森林模型"""
        results = {}
        
        for prop_name in self.property_names:
            print(f"\n训练 {prop_name} 预测模型...")
            
            # 准备数据
            y_prop = y[prop_name]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_prop, test_size=0.2, random_state=42
            )
        
            # 随机森林模型
            model = RandomForestRegressor(
                n_estimators=200,  # 增加树的数量
                random_state=42,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=2,
                n_jobs=-1  # 使用所有CPU核心
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估指标
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # 保存模型
            self.models[prop_name] = model
            
            # 保存模型信息和性能
            self.model_info[prop_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'feature_importances': model.feature_importances_
            }
            
            # 保存结果
            results[prop_name] = {
                'model': 'RandomForest',
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"RandomForest - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
        return results
    
    def save_models(self, model_dir='models'):
        """保存所有训练好的模型"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        for prop_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{prop_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"已保存 {prop_name} 模型到 {model_path}")
        
        # 保存模型信息和特征名称
        info_path = os.path.join(model_dir, 'model_info.joblib')
        info = {
            'model_info': self.model_info,
            'feature_names': self.feature_names,
            'property_names': self.property_names
        }
        joblib.dump(info, info_path)
        print(f"已保存模型信息到 {info_path}")
    
    def load_models(self, model_dir='models'):
        """加载已保存的模型"""
        for prop_name in self.property_names:
            model_path = os.path.join(model_dir, f'{prop_name}_model.joblib')
            if os.path.exists(model_path):
                self.models[prop_name] = joblib.load(model_path)
                print(f"已加载 {prop_name} 模型")
            else:
                print(f"警告: {model_path} 不存在")
        
        # 加载模型信息
        info_path = os.path.join(model_dir, 'model_info.joblib')
        if os.path.exists(info_path):
            info = joblib.load(info_path)
            self.model_info = info.get('model_info', {})
            self.feature_names = info.get('feature_names', self.feature_names)
            self.property_names = info.get('property_names', self.property_names)
            print("已加载模型信息")
    
    def predict_properties(self, composition, temperature):
        """
        预测给定配比和温度下的热物性参数
        """
        # 确保成分总和为100%
        composition = np.array(composition)
        if np.sum(composition) != 100:
            composition = composition / np.sum(composition) * 100
        
        # 准备输入特征
        input_features = np.array([[*composition, temperature]])
        
        predictions = {}
        
        for prop_name in self.property_names:
            if prop_name in self.models:
                model = self.models[prop_name]
                prediction = model.predict(input_features)[0]
                predictions[prop_name] = prediction
            else:
                predictions[prop_name] = None
        
        return predictions

def main():
    """主函数：训练并保存模型"""
    # 初始化预测器
    predictor = MoltenSaltPropertyPredictor()

    # 从Excel加载训练数据
    print("从Excel加载训练数据...")
    file_path = r'D:\project\data\NKM1.xlsx'  # 修改为你的数据文件路径
    X, y = predictor.load_data(file_path)
    
    # 训练模型
    print("训练随机森林模型...")
    results = predictor.train_models(X, y)
    
    # 打印模型性能
    print("\n=== 模型性能总结 ===")
    for prop_name, result in results.items():
        print(f"{prop_name}:")
        print(f"  R²: {result['r2']:.4f}, MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
    
    # 保存模型
    print("\n保存模型...")
    predictor.save_models('models')
    
    # 示例预测
    print("\n=== 示例预测 ===")
    example_composition = [21, 41, 38]  # NaCl 21%, KCl 41%, MgCl2 38%
    example_temperature = 900  # K
    
    predictions = predictor.predict_properties(example_composition, example_temperature)
    
    print(f"成分: NaCl {example_composition[0]}%, KCl {example_composition[1]}%, MgCl2 {example_composition[2]}%")
    print(f"温度: {example_temperature}K")
    for prop_name, value in predictions.items():
        print(f"{prop_name}: {value:.4f}")
    
    print("\n模型训练和保存完成！")

if __name__ == "__main__":
    main()