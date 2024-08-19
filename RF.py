import streamlit as st
import numpy as np
from joblib import load

# 加载模型
model = load('RF-3.pkl') # 请确保路径正确，且模型文件名为your_model.joblib


# 转换结果的标签
def convert_label(label):
    return 'normal' if label == 0 else 'RB'

def main():
    st.title('RF model screening for retinoblastoma (RB)')
    st.write('Please enter the following indicators to screen:')

    # 创建输入框接收特征值
    feature1 = st.number_input('Basophil%')
    feature2 = st.number_input('Lymphocyte')
    feature3 = st.number_input('RDWCV')

    # 当所有特征值都已输入时，激活预测按钮
    if feature1 is not None and feature2 is not None and feature3 is not None:
        # 创建一个按钮，点击后进行预测
        if st.button('screening'):
            # 将输入的特征值组合成numpy数组
            input_data = np.array([[feature1, feature2, feature3]]).astype(float)

            # 使用模型进行预测
            predictions = model.predict(input_data)
            predicted_class = predictions[0]

            # 提取预测概率
            probabilities = model.predict_proba(input_data)[0]

            # 显示预测结果
            result_label = convert_label(predicted_class)
            st.write('Screening result：', result_label)

            # 显示预测概率
            st.write('The probability of screening as {}：{:.2f}%'.format('normal' if predicted_class == 0 else 'RB',
                                                       probabilities[predicted_class] * 100))
    else:
        st.write('Please fill in all feature values!')


if __name__ == '__main__':
    main()
