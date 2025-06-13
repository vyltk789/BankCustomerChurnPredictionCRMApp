import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense

# ------------------ LOAD & PREPROCESS ------------------
@st.cache_data
def load_and_preprocess():
    df_org = pd.read_csv("Bank Customer Churn Prediction.csv")
    df = pd.read_csv("Bank Customer Churn Prediction.csv")
    df = df.drop(['customer_id'], axis=1)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('churn', axis=1)
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    return df_org, df, X, y, X_train_res, X_test_scaled, y_train_res, y_test, X_test, scaler, X.columns, X_test


# ------------------ BUILD MODEL ------------------
def build_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@st.cache_resource
def train_ann_model(X_train, y_train):
    model = build_ann_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    return model


# ------------------ EVALUATE OTHER MODELS ------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, preds), 3),
        'Precision': round(precision_score(y_test, preds), 3),
        'Recall': round(recall_score(y_test, preds), 3),
        'F1-Score': round(f1_score(y_test, preds), 3),
        'ROC-AUC': round(roc_auc_score(y_test, preds), 3)
    }


# ------------------ ACTION RECOMMENDER ------------------
def suggest_action(score):
    if score >= 70:
        return "Gọi điện và đề nghị ưu đãi"
    elif score >= 30 and score < 70:
        return "Gửi email chăm sóc"
    else:
        return "Tiếp tục theo dõi"


# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="CRM Churn Dashboard", layout="wide")
st.title("🔮 Dự đoán & Gợi ý hành động giảm rời bỏ khách hàng ngân hàng")
tabs = st.tabs(["📊 Khám phá dữ liệu", "🔮 Dự đoán churn", "📋 So sánh mô hình", "🔍 SHAP Giải thích"])

# ---------------- TAB 1: EXPLORATION ----------------
with tabs[0]:
    st.title("📊 Khám phá dữ liệu")
    df_org, df, X, y, *_ = load_and_preprocess()

    st.subheader("👁️ Tổng quan")
    st.dataframe(df_org.head(10))

    st.subheader("⚖️ Phân phối churn")
    fig = plt.figure(figsize=(5, 3))
    sns.countplot(x='churn', data=df)
    st.pyplot(fig)

    # Tương quan các đặc trưng
    st.subheader("🔗 Ma trận tương quan")
    fig_corr = plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
    st.pyplot(fig_corr)

    st.subheader("📈 Phân tích đặc trưng")
    col = st.selectbox("Chọn đặc trưng", df.columns.drop("churn"))
    fig2 = plt.figure()
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(data=df, x=col, hue="churn", kde=True)
    else:
        sns.countplot(data=df, x=col, hue="churn")
    st.pyplot(fig2)

# ---------------- TAB 2: PREDICTION ----------------
with tabs[1]:
    st.title("🔮 Dự đoán churn & Gợi ý hành động")

    _, _, _, _, X_train, X_test_scaled, y_train, y_test, X_test_raw, scaler, feature_names, X_test_unscaled = load_and_preprocess()

    # Build model
    model = train_ann_model(X_train, y_train)

    probs = model.predict(X_test_scaled).flatten()
    preds = (probs >= 0.5).astype(int)

    results_df = X_test_raw.copy()
    results_df['Risk_Score'] = np.round(probs * 100, 2)
    results_df['Prediction'] = preds
    results_df['CRM_Action'] = results_df['Risk_Score'].apply(suggest_action)

    st.subheader("📋 Kết quả CRM & hành động")
    st.dataframe(results_df[['Risk_Score', 'Prediction', 'CRM_Action']].head(20))

    ### Xem chi tiết
    st.subheader("🔎 Xem chi tiết từng khách hàng")
    selected_idx = st.selectbox("Chọn dòng khách hàng", results_df.index)
    st.write(results_df.loc[selected_idx])

    st.subheader("🎯 Lọc khách hàng rủi ro cao")
    thresh = st.slider("Ngưỡng rủi ro (%)", 0, 100, 70)
    high_risk = results_df[results_df['Risk_Score'] >= thresh]
    st.write(f"Số khách hàng cần ưu tiên: {len(high_risk)}")
    st.dataframe(high_risk[['Risk_Score', 'CRM_Action']])

# ---------------- TAB 3: MODEL COMPARISON ----------------
with tabs[2]:
    st.title("📋 So sánh mô hình: ANN vs Logistic vs Random Forest")

    _, _, _, _, X_train, X_test_scaled, y_train, y_test, _, _, _, _ = load_and_preprocess()

    ann = train_ann_model(X_train, y_train)
    ann_preds = (ann.predict(X_test_scaled) >= 0.5).astype(int)

    lr = LogisticRegression().fit(X_train, y_train)
    rf = RandomForestClassifier().fit(X_train, y_train)

    lr_preds = lr.predict(X_test_scaled)
    rf_preds = rf.predict(X_test_scaled)

    # Tổng hợp result
    models_results = [
        evaluate_model("Logistic Regression", lr, X_train, y_train, X_test_scaled, y_test),
        evaluate_model("Random Forest", rf, X_train, y_train, X_test_scaled, y_test),
        {
            'Model': 'ANN (Keras)',
            'Accuracy': round(accuracy_score(y_test, ann_preds), 3),
            'Precision': round(precision_score(y_test, ann_preds), 3),
            'Recall': round(recall_score(y_test, ann_preds), 3),
            'F1-Score': round(f1_score(y_test, ann_preds), 3),
            'ROC-AUC': round(roc_auc_score(y_test, ann_preds), 3)
        }
    ]

    st.dataframe(pd.DataFrame(models_results), use_container_width=True)

    st.title("📋 Kết quả chi tiết từng mô hình (Classification Report):")

    # Tổng hợp report
    reports = {
        "ANN": classification_report(y_test, ann_preds, output_dict=True),
        "Logistic": classification_report(y_test, lr_preds, output_dict=True),
        "Random Forest": classification_report(y_test, rf_preds, output_dict=True),
    }

    # Hiển thị
    for name, rep in reports.items():
        st.subheader(f"📈 {name}")
        st.dataframe(pd.DataFrame(rep).transpose().round(2))

# ---------------- TAB 4: SHAP ----------------
with tabs[3]:
    st.title("🔍 Giải thích mô hình ANN với SHAP (Top 100 khách hàng)")
    _, _, _, _, X_train, X_test_scaled, y_train, y_test, _, _, _, _ = load_and_preprocess()
    model = train_ann_model(X_train, y_train)

    X_for_shap = X_test_scaled[:100]
    columns = X.columns  # ['credit_score', 'country', ..., 'estimated_salary']

    # Tạo DeepExplainer
    explainer = shap.DeepExplainer(model, X_for_shap)
    shap_values = explainer.shap_values(X_for_shap)

    # Chuyển từ shape (100, 10, 1) → (100, 10)
    shap_values_2d = np.squeeze(shap_values)

    # Tạo DataFrame để vẽ biểu đồ
    X_shap_df = pd.DataFrame(X_for_shap, columns=columns)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values_2d, features=X_shap_df, feature_names=columns)
    st.pyplot(fig_shap)
