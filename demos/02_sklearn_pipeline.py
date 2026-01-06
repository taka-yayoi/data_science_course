# Databricks notebook source
# MAGIC %md
# MAGIC # デモ2: scikit-learnによるモデル開発
# MAGIC 
# MAGIC このノートブックでは、scikit-learnを使ってワインの品質を予測するモデルを構築します。
# MAGIC 
# MAGIC ## 学習目標
# MAGIC - scikit-learnの基本的なワークフローを理解する
# MAGIC - 前処理パイプラインの構築方法を学ぶ
# MAGIC - モデルの学習と評価を行う

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. データの準備

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# COMMAND ----------

# Wine Qualityデータセットの読み込み
wine_spark_df = spark.read.csv(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

# pandasに変換
df = wine_spark_df.toPandas()
print(f"データサイズ: {df.shape}")
df.head()

# COMMAND ----------

# カラム名の確認
print("カラム名:")
print(df.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 特徴量とラベルの定義

# COMMAND ----------

# 品質スコアを二値分類に変換(6以上を「高品質」とする)
df["label"] = (df["quality"] >= 6).astype(int)

# ラベルの分布を確認
print("ラベルの分布:")
print(df["label"].value_counts())
print(f"\n高品質の割合: {df['label'].mean():.2%}")

# COMMAND ----------

# 特徴量とターゲットの分離
feature_cols = [c for c in df.columns if c not in ["quality", "label"]]
print(f"特徴量: {feature_cols}")

X = df[feature_cols]
y = df["label"]

print(f"\n特徴量の形状: {X.shape}")
print(f"ターゲットの形状: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 訓練データとテストデータの分割

# COMMAND ----------

# データを訓練用(80%)とテスト用(20%)に分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"訓練データ: {X_train.shape[0]} 件")
print(f"テストデータ: {X_test.shape[0]} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. scikit-learn Pipelineの構築

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 パイプラインの概念
# MAGIC 
# MAGIC scikit-learnのPipelineは、複数の処理ステップを連結します:
# MAGIC 
# MAGIC ```
# MAGIC [StandardScaler] → [LogisticRegression]
# MAGIC      ↓                    ↓
# MAGIC   前処理              モデル学習
# MAGIC ```
# MAGIC 
# MAGIC パイプラインの利点:
# MAGIC - 前処理とモデル学習を一貫して管理
# MAGIC - データリークを防止(テストデータに訓練データの情報が漏れない)
# MAGIC - モデルの保存・読み込みが容易

# COMMAND ----------

# パイプラインの構築
pipeline = Pipeline([
    ("scaler", StandardScaler()),           # Step 1: 標準化
    ("classifier", LogisticRegression(      # Step 2: ロジスティック回帰
        max_iter=100,
        random_state=42
    ))
])

print("Pipeline構成:")
for name, step in pipeline.named_steps.items():
    print(f"  - {name}: {type(step).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 パイプラインの学習

# COMMAND ----------

# パイプラインの学習(fit)
# 内部的に:
# 1. StandardScalerがX_trainで平均・標準偏差を学習
# 2. 標準化されたデータでLogisticRegressionを学習
pipeline.fit(X_train, y_train)

print("モデルの学習が完了しました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. モデルの評価

# COMMAND ----------

# テストデータで予測
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# COMMAND ----------

# 評価指標の計算
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("=" * 40)
print("モデル評価結果")
print("=" * 40)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC:  {auc:.4f}")

# COMMAND ----------

# 分類レポート
print("\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=["標準品質", "高品質"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 混同行列の確認

# COMMAND ----------

# 混同行列
cm = confusion_matrix(y_test, y_pred)
print("混同行列:")
print(cm)

# COMMAND ----------

# 混同行列を可視化
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["標準品質", "高品質"])
ax.set_yticklabels(["標準品質", "高品質"])
ax.set_xlabel("予測")
ax.set_ylabel("実際")
ax.set_title("混同行列")

# 数値を表示
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16)

plt.colorbar(im)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. モデルの係数確認

# COMMAND ----------

# ロジスティック回帰モデルの係数を取得
lr_model = pipeline.named_steps["classifier"]

print("モデルの係数:")
print(f"切片: {lr_model.intercept_[0]:.4f}")
print("\n特徴量の係数:")
for feature, coef in sorted(zip(feature_cols, lr_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feature}: {coef:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 新しいデータでの予測

# COMMAND ----------

# テストデータから1件取得して予測
new_data = X_test.iloc[[0]]
print("入力データ:")
print(new_data)

# COMMAND ----------

# 予測の実行
prediction = pipeline.predict(new_data)
probability = pipeline.predict_proba(new_data)

print(f"\n予測結果: {'高品質' if prediction[0] == 1 else '標準品質'}")
print(f"確率: 標準品質={probability[0][0]:.2%}, 高品質={probability[0][1]:.2%}")
print(f"実際のラベル: {'高品質' if y_test.iloc[0] == 1 else '標準品質'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 
# MAGIC このデモで学んだ内容:
# MAGIC 
# MAGIC 1. **StandardScaler**: 特徴量の標準化(平均0、標準偏差1)
# MAGIC 2. **LogisticRegression**: 二値分類モデル
# MAGIC 3. **Pipeline**: 複数のステップを連結してワークフローを管理
# MAGIC 4. **評価指標**: Accuracy、F1 Score、AUC-ROCによるモデル評価
# MAGIC 
# MAGIC ### Pipelineの利点
# MAGIC 
# MAGIC | 利点 | 説明 |
# MAGIC |------|------|
# MAGIC | 一貫性 | 前処理からモデル学習まで一貫したワークフロー |
# MAGIC | データリーク防止 | テストデータの情報が訓練に漏れない |
# MAGIC | 保存容易 | パイプライン全体を1つのオブジェクトとして保存 |
# MAGIC | 再利用性 | 新しいデータに同じ処理を簡単に適用 |
# MAGIC 
# MAGIC 次のデモでは、MLflowを使った実験管理を行います。
