import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 设置页面配置
st.set_page_config(
    page_title="熔盐热物性预测系统",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF6B35;
    }
    .property-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .range-warning {
        color: #FF6B35;
        font-weight: bold;
        background-color: #FFF5F5;
        padding: 5px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class MoltenSaltPredictorApp:
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.feature_names = ['NaCl', 'KCl', 'MgCl2', 'temperature']
        self.property_names = ['density', 'heat_capacity', 'viscosity', 'thermal_conductivity']
        self.property_units = {
            'density': 'kg/m³',
            'heat_capacity': 'J/(kg·K)',
            'viscosity': 'mPa·s',
            'thermal_conductivity': 'W/(m·K)'
        }
        self.property_full_names = {
            'density': '密度',
            'heat_capacity': '比热容',
            'viscosity': '粘度',
            'thermal_conductivity': '热导率'
        }
        
    def load_models(self):
        """加载已训练的模型"""
        try:
            # 加载每个属性的模型
            for prop_name in self.property_names:
                model_path = f'models/{prop_name}_model.joblib'
                if os.path.exists(model_path):
                    self.models[prop_name] = joblib.load(model_path)
            
            # 加载模型信息
            info_path = 'models/model_info.joblib'
            if os.path.exists(info_path):
                info = joblib.load(info_path)
                self.model_info = info.get('model_info', {})
                self.feature_names = info.get('feature_names', self.feature_names)
                self.property_names = info.get('property_names', self.property_names)
            
            return True
        except Exception as e:
            st.error(f"加载模型时出错: {str(e)}")
            return False
    
    def predict(self, composition, temperature):
        """进行预测"""
        # 确保成分总和为100%
        composition = np.array(composition)
        if np.sum(composition) != 100:
            composition = composition / np.sum(composition) * 100
        
        # 准备输入特征
        input_features = np.array([[*composition, temperature]])
        
        predictions = {}
        uncertainties = {}
        
        for prop_name, model in self.models.items():
            try:
                # 预测
                prediction = model.predict(input_features)[0]
                predictions[prop_name] = prediction
                
                # 计算不确定性（基于模型性能）
                if prop_name in self.model_info:
                    rmse = self.model_info[prop_name].get('rmse', 0)
                    uncertainties[prop_name] = rmse
            except Exception as e:
                st.error(f"预测{prop_name}时出错: {str(e)}")
                predictions[prop_name] = None
        
        return predictions, uncertainties
    
    def validate_input(self, nacl, kcl, mgcl2, temperature):
        """验证输入范围"""
        warnings = []
        
        # 检查成分范围
        ranges = {
            'NaCl': (20, 35),
            'KCl': (15, 35),
            'MgCl2': (35, 50)
        }
        
        components = {
            'NaCl': nacl,
            'KCl': kcl,
            'MgCl2': mgcl2
        }
        
        total = nacl + kcl + mgcl2
        
        for name, value in components.items():
            min_val, max_val = ranges[name]
            if value < min_val or value > max_val:
                warnings.append(f"{name}: {value}% 超出推荐范围({min_val}-{max_val}%)")
        
        if abs(total - 100) > 0.1:  # 允许微小误差
            warnings.append(f"成分总和为{total:.1f}%，应为100%")
        
        # 检查温度范围
        if temperature < 700 or temperature > 1100:
            warnings.append(f"温度{temperature}K超出推荐范围(700-1100K)")
        
        return warnings
    
    def create_composition_chart(self, nacl, kcl, mgcl2):
        """创建成分饼图"""
        labels = ['NaCl', 'KCl', 'MgCl2']
        values = [nacl, kcl, mgcl2]
        colors = ['#FF6B35', '#004E89', '#00A896']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="熔盐成分分布",
            title_x=0.5,
            showlegend=True,
            height=300
        )
        
        return fig
    
    def create_property_chart(self, predictions):
        """创建物性参数柱状图"""
        properties = list(predictions.keys())
        values = list(predictions.values())
        units = [self.property_units.get(p, '') for p in properties]
        full_names = [self.property_full_names.get(p, p) for p in properties]
        
        colors = ['#FF6B35', '#004E89', '#00A896', '#F9C74F']
        
        fig = go.Figure(data=[go.Bar(
            x=full_names,
            y=values,
            text=[f'{v:.2f} {u}' for v, u in zip(values, units)],
            textposition='auto',
            marker_color=colors,
            hovertext=[f'{full_names[i]}: {values[i]:.2f} {units[i]}' for i in range(len(values))]
        )])
        
        fig.update_layout(
            title="预测的热物性参数",
            title_x=0.5,
            xaxis_title="物性参数",
            yaxis_title="数值",
            height=400
        )
        
        return fig
    
    def create_temperature_sensitivity_plot(self, base_composition, base_temp):
        """创建温度敏感性分析图"""
        temps = np.linspace(700, 1100, 20)
        predictions_temp = {}
        
        for prop_name in self.property_names:
            if prop_name in self.models:
                preds = []
                for temp in temps:
                    input_features = np.array([[*base_composition, temp]])
                    pred = self.models[prop_name].predict(input_features)[0]
                    preds.append(pred)
                predictions_temp[prop_name] = preds
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[self.property_full_names.get(p, p) for p in self.property_names]
        )
        
        colors = ['#FF6B35', '#004E896', '#00A896', '#F9C74F']
        
        for idx, prop_name in enumerate(self.property_names):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            if prop_name in predictions_temp:
                fig.add_trace(
                    go.Scatter(
                        x=temps,
                        y=predictions_temp[prop_name],
                        mode='lines+markers',
                        name=self.property_full_names.get(prop_name, prop_name),
                        line=dict(color=colors[idx], width=2),
                        marker=dict(size=6)
                    ),
                    row=row, col=col
                )
                
                # 添加基准温度标记
                if prop_name in predictions_temp:
                    base_idx = np.abs(temps - base_temp).argmin()
                    base_value = predictions_temp[prop_name][base_idx]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[base_temp],
                            y=[base_value],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='diamond'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=f"温度敏感性分析 (基准成分: {base_composition[0]:.1f}-{base_composition[1]:.1f}-{base_composition[2]:.1f}%)",
            title_x=0.5
        )
        
        return fig
    
    def run(self):
        """运行Streamlit应用"""
        # 应用标题
        st.markdown('<h1 class="main-header">🔥 NaCl-KCl-MgCl2熔盐热物性预测系统</h1>', unsafe_allow_html=True)
        st.markdown("### 基于机器学习的熔盐热物性参数智能预测")
        
        # 加载模型
        with st.spinner('正在加载预测模型...'):
            if not self.load_models():
                st.error("无法加载模型，请确保模型文件存在")
                return
        
        # 创建两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">📝 输入参数</div>', unsafe_allow_html=True)
            with st.form("prediction_form"):
                st.markdown("#### 熔盐成分 (mol%)")
                nacl = st.slider("NaCl 含量", 0.0, 100.0, 25.0, 0.5, help="推荐范围: 20-35%")
                kcl = st.slider("KCl 含量", 0.0, 100.0, 30.0, 0.5, help="推荐范围: 15-35%")
                mgcl2 = st.slider("MgCl₂ 含量", 0.0, 100.0, 45.0, 0.5, help="推荐范围: 35-50%")
                total = nacl + kcl + mgcl2
                st.metric("成分总和", f"{total:.1f}%", delta=f"{total-100:.1f}%" if total != 100 else None)
                if abs(total - 100) > 0.1:
                    st.warning(f"⚠️ 成分总和为 {total:.1f}%，必须为100%才能预测")
                
                st.markdown("#### 温度参数")
                temperature = st.slider("温度 (K)", 600, 1200, 900, 10, help="推荐范围: 700-1100K")
                
                submit_button = st.form_submit_button("🚀 开始预测", use_container_width=True)
        
        with col2:
            # 饼图只在点击后显示
            if submit_button:
                fig_composition = self.create_composition_chart(nacl, kcl, mgcl2)
                st.plotly_chart(fig_composition, use_container_width=True)
            else:
                st.info("调整左侧参数后，点击'开始预测'按钮")

        # 验证输入范围（仅警告，不阻止预测）
        warnings = self.validate_input(nacl, kcl, mgcl2, temperature)
        if warnings:
            for warning in warnings:
                st.markdown(f'<div class="range-warning">⚠️ {warning}</div>', unsafe_allow_html=True)

        # 预测逻辑
        if submit_button:
            # 新增：检查总和是否为100%
            if abs(total - 100) > 0.1:
                st.error("❌ 成分总和必须等于100%才能进行预测。请调整各成分含量。")
            else:
                with st.spinner('正在进行预测计算...'):
                    normalized_comp = [nacl, kcl, mgcl2]  # 总和已为100%，无需归一化
                    predictions, uncertainties = self.predict(normalized_comp, temperature)
                    
                    if predictions:
                        st.markdown('<div class="sub-header">📊 预测结果</div>', unsafe_allow_html=True)

                    
                    # 显示预测结果
                    cols = st.columns(4)
                    property_display_names = {
                        'density': '密度',
                        'heat_capacity': '比热容',
                        'viscosity': '粘度',
                        'thermal_conductivity': '热导率'
                    }
                    
                    property_icons = {
                        'density': '⚖️',
                        'heat_capacity': '🔥',
                        'viscosity': '💧',
                        'thermal_conductivity': '📈'
                    }
                    
                    for idx, (prop_name, value) in enumerate(predictions.items()):
                        with cols[idx]:
                            unit = self.property_units.get(prop_name, '')
                            icon = property_icons.get(prop_name, '📊')
                            display_name = property_display_names.get(prop_name, prop_name)
                            
                            if value is not None:
                                st.markdown(f'<div class="property-card">', unsafe_allow_html=True)
                                st.metric(
                                    f"{icon} {display_name}",
                                    f"{value:.2f} {unit}",
                                    delta=f"±{uncertainties.get(prop_name, 0):.2f}" if prop_name in uncertainties else None
                                )
                                st.markdown(f'</div>', unsafe_allow_html=True)
                    
        
        # 侧边栏信息
        with st.sidebar:
            st.markdown("## ℹ️ 系统说明")
            st.info("""
            **系统功能：**
            - 预测NaCl-KCl-MgCl2熔盐的热物性参数
            - 支持成分配比和温度调节
            - 提供温度敏感性分析
            
            **输入范围建议：**
            - MgCl₂: 35–50 mol%
            - KCl: 15–35 mol%
            - NaCl: 20–35 mol%
            - 温度: 700-1100 K
            
            **预测参数：**
            1. 密度 (kg/m³)
            2. 比热容 (J/(kg·K))
            3. 粘度 (mPa·s)
            4. 热导率 (W/(m·K))
            """)
            
            # 示例配比
            st.markdown("## 💡 示例配比")
            if st.button("示例1: 标准配比"):
                st.session_state.nacl = 25.0
                st.session_state.kcl = 30.0
                st.session_state.mgcl2 = 45.0
                st.session_state.temperature = 900
                st.rerun()
            
            if st.button("示例2: 高MgCl₂"):
                st.session_state.nacl = 20.0
                st.session_state.kcl = 25.0
                st.session_state.mgcl2 = 55.0
                st.session_state.temperature = 850
                st.rerun()

# 运行应用
if __name__ == "__main__":
    app = MoltenSaltPredictorApp()
    app.run()