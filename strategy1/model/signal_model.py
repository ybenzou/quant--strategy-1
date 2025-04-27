import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap


def build_labels(df, threshold=0.01):
    """
    简单基于未来一天收益构建标签：
    - 涨幅超过 +1% -> 买入 (1)
    - 跌幅超过 -1% -> 卖出 (-1)
    - 其余 -> 持有 (0)
    """
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    labels = np.where(future_return > threshold, 1, 
                      np.where(future_return < -threshold, -1, 0))
    return labels

def train_signal_model(data_dict, save_model_path="models/trading_model.pkl"):
    """
    给定所有股票的数据，训练一个交易信号分类器。
    """
    all_features = []
    all_labels = []

    for ticker, df in data_dict.items():
        df = df.dropna()

        # 特征
        features = df[["Return", "Volume_Change", "MA5", "MA20", "MA5_minus_MA20", "Volatility10", "RSI14"]]
        features = features.shift(1).dropna()  # 只用前一天的特征来预测今天的动作

        # 标签
        labels = build_labels(df).reshape(-1, )
        labels = pd.Series(labels, index=df.index)
        labels = labels.loc[features.index]

        all_features.append(features)
        all_labels.append(labels)

    X = pd.concat(all_features)
    y = pd.concat(all_labels)

    # 划分训练集/测试集（这里简单切分，后面可以做更专业的时序验证）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    joblib.dump(model, save_model_path)
    print(f"[INFO] Model saved to {save_model_path}")


def predict_signals(model_path, latest_data):
    model = joblib.load(model_path)
    signals = {}

    feature_names = ["Return", "Volume_Change", "MA5", "MA20", "MA5_minus_MA20", "Volatility10", "RSI14"]

    for ticker, df in latest_data.items():
        df = df.dropna()
        if df.shape[0] < 2:
            continue

        latest_features = df[feature_names].iloc[-2]
        X_latest = latest_features.values.reshape(1, -1)

        pred_prob = model.predict_proba(X_latest)
        pred_label = model.predict(X_latest)[0]
        confidence = np.max(pred_prob)

        # 🔥 每个股票单独保存一张解释图
        explain_single_prediction(model, feature_names, X_latest, ticker)

        # 仓位建议
        if confidence > 0.9:
            position = 0.8
        elif confidence > 0.7:
            position = 0.5
        elif confidence > 0.6:
            position = 0.3
        else:
            position = 0.0

        signals[ticker] = (pred_label, confidence, position)

    return signals


def plot_feature_importance(model, feature_names, save_path="plots/feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title("Feature Importance in Trading Signal Model")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # ✅ 不展示，只保存！
    print(f"[INFO] Feature importance plot saved to {save_path}")


def explain_single_prediction(model, feature_names, X_latest, ticker, save_dir="plots/explanations"):
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_latest)

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(shap_values, list):
        pred_class = model.predict(X_latest)[0]
        shap_value = shap_values[pred_class][0]
    else:
        shap_value = shap_values[0]

    shap_value = shap_value.flatten()
    n_shap_features = len(shap_value)

    if len(feature_names) == n_shap_features:
        current_feature_names = feature_names
    else:
        current_feature_names = [f"Feature {i}" for i in range(n_shap_features)]

    sorted_idx = np.argsort(np.abs(shap_value))[::-1]
    sorted_features = [current_feature_names[i] for i in sorted_idx]
    sorted_values = shap_value[sorted_idx]

    # 1. 找到主驱动特征
    main_feature = sorted_features[0]
    main_contribution = sorted_values[0]

    # 2. 绘制图，高亮主驱动
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if i == 0 else 'gray' for i in range(len(sorted_features))]  # 主驱动用红色
    bars = ax.bar(range(len(sorted_features)), sorted_values, color=colors)
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax.set_title(f"Feature contributions for {ticker}")
    ax.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{ticker}_explanation.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Explanation plot saved to {save_path}")

    return main_feature, main_contribution
