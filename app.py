"""
Streamlit 垃圾郵件分類應用程式
提供互動式 UI 用於訓練模型、評估性能和即時預測
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append(str(Path(__file__).parent))

from src.data_preprocessing import load_data, preprocess_data, create_vectorizer, prepare_features
from src.models import SpamClassifier, train_all_models
from src.evaluation import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curve,
    generate_classification_report, compare_models, plot_model_comparison
)

# 設定頁面配置
st.set_page_config(
    page_title="垃圾郵件分類系統",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 初始化 session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'df' not in st.session_state:
    st.session_state.df = None


def load_dataset():
    """載入資料集"""
    try:
        df = load_data('datasets/sms_spam_no_header.csv')
        df_processed = preprocess_data(df)
        st.session_state.df = df
        st.session_state.df_processed = df_processed
        return True
    except Exception as e:
        st.error(f"載入資料集失敗: {str(e)}")
        return False


def main():
    """主函數"""
    st.title("📧 垃圾郵件分類系統")
    st.markdown("---")
    
    # 側邊欄
    st.sidebar.title("導航")
    page = st.sidebar.radio(
        "選擇頁面",
        ["資料集概覽", "模型訓練", "模型評估", "即時預測", "模型比較"]
    )
    
    # 載入資料集
    if st.session_state.df is None:
        with st.spinner("載入資料集..."):
            if not load_dataset():
                st.stop()
    
    # 根據選擇的頁面顯示內容
    if page == "資料集概覽":
        show_dataset_overview()
    elif page == "模型訓練":
        show_model_training()
    elif page == "模型評估":
        show_model_evaluation()
    elif page == "即時預測":
        show_realtime_prediction()
    elif page == "模型比較":
        show_model_comparison()


def show_dataset_overview():
    """顯示資料集概覽"""
    st.header("📊 資料集概覽")
    
    df = st.session_state.df
    
    # 基本統計
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總訊息數", len(df))
    with col2:
        st.metric("Ham 訊息數", len(df[df['label'] == 'ham']))
    with col3:
        st.metric("Spam 訊息數", len(df[df['label'] == 'spam']))
    with col4:
        spam_ratio = len(df[df['label'] == 'spam']) / len(df) * 100
        st.metric("Spam 比例", f"{spam_ratio:.2f}%")
    
    st.markdown("---")
    
    # 標籤分布圖
    st.subheader("標籤分布")
    fig, ax = plt.subplots(figsize=(8, 6))
    label_counts = df['label'].value_counts()
    ax.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('標籤分布')
    st.pyplot(fig)
    
    # 樣本訊息
    st.subheader("樣本訊息")
    sample_size = st.slider("顯示樣本數", 5, 50, 10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Ham 樣本:**")
        ham_samples = df[df['label'] == 'ham'].head(sample_size)
        st.dataframe(ham_samples[['label', 'text']], use_container_width=True)
    
    with col2:
        st.write("**Spam 樣本:**")
        spam_samples = df[df['label'] == 'spam'].head(sample_size)
        st.dataframe(spam_samples[['label', 'text']], use_container_width=True)


def show_model_training():
    """顯示模型訓練頁面"""
    st.header("🚀 模型訓練")
    
    st.info("點擊下方按鈕開始訓練所有模型（邏輯迴歸、樸素貝葉斯、支援向量機）")
    
    if st.button("開始訓練模型", type="primary"):
        with st.spinner("訓練模型中，請稍候..."):
            try:
                # 訓練所有模型
                models, X_test, y_test = train_all_models()
                
                # 保存到 session state
                st.session_state.models = models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                st.success("✅ 所有模型訓練完成！")
                
                # 顯示訓練結果摘要
                st.subheader("訓練結果摘要")
                for model_name, classifier in models.items():
                    st.write(f"**{model_name}**: 訓練完成")
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
    
    # 顯示已訓練的模型
    if st.session_state.models:
        st.subheader("已訓練的模型")
        for model_name in st.session_state.models.keys():
            st.success(f"✅ {model_name}")


def show_model_evaluation():
    """顯示模型評估頁面"""
    st.header("📈 模型評估")
    
    if not st.session_state.models:
        st.warning("⚠️ 請先訓練模型！")
        return
    
    # 選擇要評估的模型
    model_names = list(st.session_state.models.keys())
    selected_model = st.selectbox("選擇模型", model_names)
    
    if st.button("評估模型", type="primary"):
        with st.spinner("評估模型中..."):
            classifier = st.session_state.models[selected_model]
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # 預測
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)[:, 1]
            
            # 計算指標
            metrics = calculate_metrics(y_test, y_pred)
            
            # 顯示指標
            st.subheader("評估指標")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("準確率", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("精確率", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("召回率", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 分數", f"{metrics['f1_score']:.4f}")
            
            # 混淆矩陣
            st.subheader("混淆矩陣")
            fig_cm = plot_confusion_matrix(y_test, y_pred, selected_model)
            st.pyplot(fig_cm)
            
            # ROC 曲線
            st.subheader("ROC 曲線")
            fig_roc, auc_score = plot_roc_curve(y_test, y_proba, selected_model)
            st.pyplot(fig_roc)
            st.metric("AUC 分數", f"{auc_score:.4f}")
            
            # 分類報告
            st.subheader("分類報告")
            report = generate_classification_report(y_test, y_pred)
            st.text(report)


def show_realtime_prediction():
    """顯示即時預測頁面"""
    st.header("🔮 即時預測")
    
    if not st.session_state.models:
        st.warning("⚠️ 請先訓練模型！")
        return
    
    # 選擇模型
    model_names = list(st.session_state.models.keys())
    selected_model = st.selectbox("選擇模型", model_names)
    
    # 輸入文本
    text_input = st.text_area("輸入訊息", height=150, placeholder="在此輸入要分類的訊息...")
    
    if st.button("預測", type="primary"):
        if not text_input.strip():
            st.warning("請輸入訊息！")
        else:
            with st.spinner("預測中..."):
                classifier = st.session_state.models[selected_model]
                
                # 預處理文本
                from src.data_preprocessing import TextPreprocessor
                preprocessor = TextPreprocessor()
                processed_text = preprocessor.preprocess(text_input)
                
                # 向量化
                text_vectorized = classifier.vectorizer.transform([processed_text])
                
                # 預測
                prediction = classifier.predict(text_vectorized)[0]
                probability = classifier.predict_proba(text_vectorized)[0]
                
                # 顯示結果
                st.subheader("預測結果")
                
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 'spam':
                        st.error(f"**預測結果: {prediction.upper()}**")
                    else:
                        st.success(f"**預測結果: {prediction.upper()}**")
                
                with col2:
                    spam_prob = probability[1] if len(probability) > 1 else probability[0]
                    ham_prob = probability[0] if len(probability) > 1 else 1 - probability[0]
                    st.metric("Spam 機率", f"{spam_prob:.4f}")
                    st.metric("Ham 機率", f"{ham_prob:.4f}")
                
                # 顯示預處理後的文本
                with st.expander("查看預處理後的文本"):
                    st.text(processed_text)


def show_model_comparison():
    """顯示模型比較頁面"""
    st.header("⚖️ 模型比較")
    
    if not st.session_state.models:
        st.warning("⚠️ 請先訓練模型！")
        return
    
    if st.button("比較所有模型", type="primary"):
        with st.spinner("比較模型中..."):
            try:
                # 比較模型
                results_df = compare_models(
                    st.session_state.models,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    save_dir='results'
                )
                
                # 顯示比較結果
                st.subheader("模型比較結果")
                st.dataframe(results_df, use_container_width=True)
                
                # 繪製比較圖表
                st.subheader("模型性能比較圖表")
                fig = plot_model_comparison(results_df, save_path='results/model_comparison.png')
                st.pyplot(fig)
                
                # 找出最佳模型
                best_model = results_df.loc[results_df['f1_score'].idxmax(), 'model']
                st.success(f"🏆 最佳模型（F1 分數）: **{best_model}**")
                
            except Exception as e:
                st.error(f"比較失敗: {str(e)}")


if __name__ == '__main__':
    main()

