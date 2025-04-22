import streamlit as st
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# 加载模型
model = load('RF-3.pkl')  # 请确保模型路径正确

# 转换结果的标签
def convert_label(label):
    return 'normal' if label == 0 else 'RB'

def main():
    st.title('RF model screening for retinoblastoma (RB) Version 2.0.0')
    st.write('Please enter the following indicators to screen:')

    # 创建输入框接收特征值
    feature1 = st.number_input('Basophil%', min_value=0.0, format="%.2f")
    feature2 = st.number_input('Lymphocyte', min_value=0.0, format="%.2f")
    feature3 = st.number_input('RDWCV', min_value=0.0, format="%.2f")

    # 当所有特征值都已输入时，激活预测按钮
    if st.button('Screening'):
        if feature1 == 0 and feature2 == 0 and feature3 == 0:
            st.warning("Please enter valid feature values!")
        else:
            # 将输入的特征值组合成numpy数组
            input_data = np.array([[feature1, feature2, feature3]]).astype(float)

            # 使用模型进行预测
            predictions = model.predict(input_data)
            predicted_class = predictions[0]

            # 提取预测概率
            probabilities = model.predict_proba(input_data)[0]

            # 显示预测结果
            result_label = convert_label(predicted_class)
            st.success(f'Screening result: **{result_label}**')

            # 显示预测概率
            st.write(f'Probability of being **{result_label}**: {probabilities[predicted_class] * 100:.2f}%')

            # 创建柱状图展示概率分布
            fig, ax = plt.subplots(figsize=(8, 5))  # 调整画布大小
            classes = ['normal', 'RB']
            colors = ['red' if x == predicted_class else 'lightgray' for x in [0, 1]]  # 红色高亮预测类别
            
            bars = ax.bar(classes, probabilities, color=colors, width=0.6)
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title('Prediction Probability Distribution', 
                        fontsize=14, 
                        pad=20,  # 标题距离图的间距
                        fontweight='bold',
                        color='black')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线

            # 在每个柱子上方显示概率值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                        height + 0.02,  # 文字略微高于柱子
                        f'{height:.1%}',
                        ha='center', 
                        va='bottom', 
                        fontsize=12,
                        fontweight='bold')

            # 显示图表
            st.pyplot(fig)

if __name__ == '__main__':
    main()
