import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM

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


# ------------------ BUILD CNN MODEL ------------------
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


# ------------------ BUILD CNN MODEL ------------------
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def train_cnn_model(X_train, y_train):
    model = build_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    return model

# ------------------ BUILD RNN (LSTM) MODEL ------------------
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def train_rnn_model(X_train, y_train):
    model = build_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    return model

# ------------------ ROC CURVE ------------------
def plot_roc_curves(models, X_test_list, y_test, labels):
    fig = plt.figure(figsize=(8, 6))

    for model, X_test, label in zip(models, X_test_list, labels):
        # Dự đoán xác suất
        probs = model.predict(X_test).ravel()
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - So sánh các mô hình')
    plt.legend(loc="lower right")
    plt.grid(True)

    return fig


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
    st.title("📋 So sánh mô hình: ANN vs CNN vs RNN")

    _, _, _, _, X_train, X_test_scaled, y_train, y_test, _, _, _, _ = load_and_preprocess()

    ann = train_ann_model(X_train, y_train)
    ann_preds = (ann.predict(X_test_scaled) >= 0.5).astype(int)

    # Chuẩn bị đầu vào cho CNN và RNN
    X_train_reshape = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshape = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    cnn = train_cnn_model(X_train_reshape, y_train)
    cnn_preds = (cnn.predict(X_test_reshape) >= 0.5).astype(int)

    rnn = train_rnn_model(X_train_reshape, y_train)
    rnn_preds = (rnn.predict(X_test_reshape) >= 0.5).astype(int)

    # Tổng hợp result
    models_results = [
        {
            'Model': 'ANN (Keras)',
            'Accuracy': round(accuracy_score(y_test, ann_preds), 3),
            'Precision': round(precision_score(y_test, ann_preds), 3),
            'Recall': round(recall_score(y_test, ann_preds), 3),
            'F1-Score': round(f1_score(y_test, ann_preds), 3),
        },
        {
            'Model': 'CNN',
            'Accuracy': round(accuracy_score(y_test, cnn_preds), 3),
            'Precision': round(precision_score(y_test, cnn_preds), 3),
            'Recall': round(recall_score(y_test, cnn_preds), 3),
            'F1-Score': round(f1_score(y_test, cnn_preds), 3),
        },
        {
            'Model': 'RNN',
            'Accuracy': round(accuracy_score(y_test, rnn_preds), 3),
            'Precision': round(precision_score(y_test, rnn_preds), 3),
            'Recall': round(recall_score(y_test, rnn_preds), 3),
            'F1-Score': round(f1_score(y_test, rnn_preds), 3),
        }
    ]

    st.dataframe(pd.DataFrame(models_results), use_container_width=True)

    st.title("📋 Kết quả chi tiết từng mô hình (Classification Report):")

    # Tổng hợp report
    reports = {
        "ANN": classification_report(y_test, ann_preds, output_dict=True),
        "CNN": classification_report(y_test, cnn_preds, output_dict=True),
        "RNN": classification_report(y_test, rnn_preds, output_dict=True),
    }

    # Hiển thị
    for name, rep in reports.items():
        st.subheader(f"📈 {name}")
        st.dataframe(pd.DataFrame(rep).transpose().round(2))

    # Biểu đồ ROC Curve - So sánh các mô hình

    fig_roc_curves = plot_roc_curves(
        models=[cnn, rnn, ann],
        X_test_list=[X_test_reshape, X_test_reshape, X_test_scaled],
        y_test=y_test,
        labels=["CNN", "RNN", "ANN"]
    )
    st.pyplot(fig_roc_curves)

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
