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
    labels = ['Ham (正常郵件)' if idx == 'ham' else 'Spam (垃圾郵件)' for idx in label_counts.index]
    colors = ['#66b3ff', '#ff9999']
    ax.pie(label_counts.values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('標籤分布', fontsize=14, pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
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
    
    # 參數調整區域
    with st.expander("⚙️ 調整參數", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("資料分割參數")
            test_size = st.slider("測試集比例", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("隨機種子", 0, 1000, 42, 1)
        
        with col2:
            st.subheader("向量化器參數")
            vectorizer_type = st.selectbox("向量化器類型", ["tfidf", "count"], index=0)
            max_features = st.number_input("最大特徵數", 1000, 10000, 5000, 500)
            min_df = st.number_input("最小文檔頻率 (min_df)", 1, 10, 2, 1)
            max_df = st.slider("最大文檔頻率 (max_df)", 0.5, 1.0, 0.95, 0.05)
            ngram_range_min = st.number_input("N-gram 最小範圍", 1, 3, 1, 1)
            ngram_range_max = st.number_input("N-gram 最大範圍", 1, 3, 2, 1)
        
        st.subheader("模型超參數")
        model_tabs = st.tabs(["邏輯迴歸", "樸素貝葉斯", "支援向量機"])
        
        lr_params = {}
        nb_params = {}
        svm_params = {}
        
        with model_tabs[0]:
            col1, col2, col3 = st.columns(3)
            with col1:
                lr_params['C'] = st.number_input("正則化強度 (C)", 0.01, 100.0, 1.0, 0.1, key='lr_C')
            with col2:
                lr_params['penalty'] = st.selectbox("正則化類型", ["l2", "l1", "elasticnet"], index=0, key='lr_penalty')
            with col3:
                lr_params['solver'] = st.selectbox("求解器", ["lbfgs", "liblinear", "sag", "saga"], index=0, key='lr_solver')
            lr_params['max_iter'] = st.number_input("最大迭代次數", 100, 5000, 1000, 100, key='lr_max_iter')
        
        with model_tabs[1]:
            nb_params['alpha'] = st.number_input("平滑參數 (alpha)", 0.1, 10.0, 1.0, 0.1, key='nb_alpha')
        
        with model_tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                svm_params['C'] = st.number_input("正則化參數 (C)", 0.01, 100.0, 1.0, 0.1, key='svm_C')
            with col2:
                svm_params['kernel'] = st.selectbox("核函數", ["linear", "rbf", "poly", "sigmoid"], index=0, key='svm_kernel')
            if svm_params['kernel'] == 'rbf':
                svm_params['gamma'] = st.selectbox("Gamma", ["scale", "auto"], index=0, key='svm_gamma')
    
    st.info("點擊下方按鈕開始訓練所有模型（邏輯迴歸、樸素貝葉斯、支援向量機）")
    
    if st.button("開始訓練模型", type="primary"):
        with st.spinner("訓練模型中，請稍候..."):
            try:
                # 準備參數
                vectorizer_params = {
                    'max_features': max_features,
                    'min_df': min_df,
                    'max_df': max_df,
                    'ngram_range': (ngram_range_min, ngram_range_max)
                }
                
                model_params_dict = {
                    'logistic_regression': lr_params,
                    'naive_bayes': nb_params,
                    'svm': svm_params
                }
                
                # 訓練所有模型
                models, X_test, y_test = train_all_models(
                    test_size=test_size,
                    random_state=random_state,
                    vectorizer_params=vectorizer_params,
                    model_params_dict=model_params_dict
                )
                
                # 保存到 session state
                st.session_state.models = models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.training_params = {
                    'test_size': test_size,
                    'random_state': random_state,
                    'vectorizer_params': vectorizer_params,
                    'model_params': model_params_dict
                }
                
                st.success("✅ 所有模型訓練完成！")
                
                # 顯示訓練結果摘要
                st.subheader("訓練結果摘要")
                col1, col2, col3 = st.columns(3)
                for idx, (model_name, classifier) in enumerate(models.items()):
                    with [col1, col2, col3][idx % 3]:
                        st.success(f"✅ **{model_name}**: 訓練完成")
                
                # 顯示使用的參數
                with st.expander("查看使用的參數"):
                    st.json(st.session_state.training_params)
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                st.exception(e)
    
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
    
    # 預測選項
    with st.expander("⚙️ 預測選項", expanded=False):
        show_preprocessed = st.checkbox("顯示預處理後的文本", value=False)
        show_probability_bar = st.checkbox("顯示機率條形圖", value=True)
        threshold = st.slider("Spam 判定閾值", 0.0, 1.0, 0.5, 0.05)
    
    # 初始化範例文本
    if 'example_text' not in st.session_state:
        st.session_state.example_text = ""
    
    # 範例訊息按鈕
    st.caption("💡 提示：點擊下方按鈕使用範例訊息")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("範例 Ham 訊息", key='example_ham'):
            st.session_state.example_text = "Hey, are you free this weekend? Let's hang out!"
            st.rerun()
    with col2:
        if st.button("範例 Spam 訊息", key='example_spam'):
            st.session_state.example_text = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!"
            st.rerun()
    with col3:
        if st.button("清空輸入", key='clear_input'):
            st.session_state.example_text = ""
            st.rerun()
    
    # 輸入文本
    text_input = st.text_area("輸入訊息", height=150, 
                              value=st.session_state.example_text,
                              placeholder="在此輸入要分類的訊息...",
                              key='text_input_area')
    
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
                
                spam_prob = probability[1] if len(probability) > 1 else probability[0]
                ham_prob = probability[0] if len(probability) > 1 else 1 - probability[0]
                
                # 根據閾值判斷
                final_prediction = 'spam' if spam_prob >= threshold else 'ham'
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if final_prediction == 'spam':
                        st.error(f"**預測結果: {final_prediction.upper()}**")
                        st.caption(f"機率: {spam_prob:.2%}")
                    else:
                        st.success(f"**預測結果: {final_prediction.upper()}**")
                        st.caption(f"機率: {ham_prob:.2%}")
                
                with col2:
                    st.metric("Spam 機率", f"{spam_prob:.4f}")
                    st.progress(spam_prob)
                
                with col3:
                    st.metric("Ham 機率", f"{ham_prob:.4f}")
                    st.progress(ham_prob)
                
                # 機率條形圖
                if show_probability_bar:
                    st.subheader("機率分布")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    categories = ['Ham', 'Spam']
                    probs = [ham_prob, spam_prob]
                    colors = ['green' if p == max(probs) else 'gray' for p in probs]
                    bars = ax.bar(categories, probs, color=colors, alpha=0.7)
                    ax.set_ylim([0, 1])
                    ax.set_ylabel('機率')
                    ax.set_title('預測機率分布')
                    ax.axhline(y=threshold, color='r', linestyle='--', label=f'閾值 ({threshold})')
                    ax.legend()
                    for i, (bar, prob) in enumerate(zip(bars, probs)):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{prob:.2%}', ha='center', va='bottom')
                    st.pyplot(fig)
                
                # 顯示預處理後的文本
                if show_preprocessed:
                    st.subheader("預處理後的文本")
                    st.code(processed_text, language='text')


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

