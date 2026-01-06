# Databricks notebook source
# MAGIC %md
# MAGIC # デモ3: MLflowによる実験トラッキング
# MAGIC 
# MAGIC このノートブックでは、MLflowを使って機械学習の実験を管理します。
# MAGIC 
# MAGIC ## 学習目標
# MAGIC - MLflow Trackingの基本を理解する
# MAGIC - パラメータ、メトリクス、モデルの記録方法を学ぶ
# MAGIC - 実験の比較と分析を行う

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 実験の設定

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# COMMAND ----------

# 現在のユーザー名を取得
username = spark.sql("SELECT current_user()").collect()[0][0]

# 実験名の設定
experiment_name = f"/Users/{username}/wine_quality_experiment"
mlflow.set_experiment(experiment_name)

print(f"実験名: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. データの準備

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

# 二値分類用のラベル作成
df["label"] = (df["quality"] >= 6).astype(int)

# 特徴量とターゲットの分離
feature_cols = [c for c in df.columns if c not in ["quality", "label"]]
X = df[feature_cols]
y = df["label"]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"訓練データ: {X_train.shape[0]} 件")
print(f"テストデータ: {X_test.shape[0]} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. MLflowによる実験記録

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 手動ロギング

# COMMAND ----------

# ハイパーパラメータの設定
params = {
    "C": 1.0,
    "max_iter": 100,
    "solver": "lbfgs"
}

# MLflowラン開始
with mlflow.start_run(run_name="logistic_regression_manual"):
    
    # パラメータの記録
    mlflow.log_params(params)
    mlflow.log_param("model_type", "LogisticRegression")
    
    # パイプライン構築
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            random_state=42
        ))
    ])
    
    # モデル学習
    pipeline.fit(X_train, y_train)
    
    # 予測
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 評価指標の計算
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # メトリクスの記録
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    # モデルの記録
    mlflow.sklearn.log_model(pipeline, "sklearn_model")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    
    run_id = mlflow.active_run().info.run_id
    print(f"\nRun ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 オートロギング(autolog)

# COMMAND ----------

# MLflowのオートロギングを有効化
mlflow.autolog()

# COMMAND ----------

# 異なるパラメータで実験
params_list = [
    {"C": 0.1, "max_iter": 100},
    {"C": 1.0, "max_iter": 200},
    {"C": 10.0, "max_iter": 100},
]

for i, params in enumerate(params_list):
    with mlflow.start_run(run_name=f"lr_experiment_{i+1}"):
        # タグの追加
        mlflow.set_tag("experiment_type", "hyperparameter_search")
        mlflow.set_tag("iteration", i+1)
        
        # パイプライン構築
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=params["C"],
                max_iter=params["max_iter"],
                random_state=42
            ))
        ])
        
        # モデル学習(autologにより自動記録)
        pipeline.fit(X_train, y_train)
        
        # 追加メトリクスの記録
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("test_auc", auc)
        
        print(f"実験 {i+1}: C={params['C']}, max_iter={params['max_iter']} -> AUC={auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 異なるアルゴリズムの比較

# COMMAND ----------

# RandomForestでも実験
with mlflow.start_run(run_name="random_forest"):
    mlflow.set_tag("model_type", "RandomForest")
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("test_auc", auc)
    
    print(f"RandomForest AUC: {auc:.4f}")

# COMMAND ----------

# オートロギングを無効化
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 実験結果の確認

# COMMAND ----------

# 実験のランを取得
experiment = mlflow.get_experiment_by_name(experiment_name)
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# 結果の表示
display(runs_df[["run_id", "metrics.test_auc", "tags.mlflow.runName", "status"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ベストモデルの特定

# COMMAND ----------

# test_aucでソートしてベストモデルを取得
runs_with_auc = runs_df[runs_df["metrics.test_auc"].notna()].copy()
best_run = runs_with_auc.loc[runs_with_auc["metrics.test_auc"].idxmax()]

print("=" * 50)
print("ベストモデル")
print("=" * 50)
print(f"Run ID: {best_run['run_id']}")
print(f"Run Name: {best_run['tags.mlflow.runName']}")
print(f"Test AUC: {best_run['metrics.test_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. モデルの読み込みと推論

# COMMAND ----------

# ベストモデルの読み込み
best_run_id = best_run["run_id"]
loaded_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/sklearn_model")

# 新しいデータで予測
sample_data = X_test.iloc[:5]
new_predictions = loaded_model.predict(sample_data)
new_probabilities = loaded_model.predict_proba(sample_data)[:, 1]

print("予測結果:")
for i, (pred, prob, actual) in enumerate(zip(new_predictions, new_probabilities, y_test.iloc[:5])):
    print(f"  サンプル{i+1}: 予測={pred}, 確率={prob:.2%}, 実際={actual}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MLflow UIへのリンク

# COMMAND ----------

# 実験へのリンクを表示
print(f"MLflow実験ページ: 左サイドバーの「実験」から確認できます")
print(f"実験名: {experiment_name}")
print(f"実験ID: {experiment.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 
# MAGIC このデモで学んだ内容:
# MAGIC 
# MAGIC ### MLflow Trackingの主要機能
# MAGIC 
# MAGIC | 機能 | 説明 | API |
# MAGIC |------|------|-----|
# MAGIC | パラメータ記録 | ハイパーパラメータを保存 | `mlflow.log_param()` |
# MAGIC | メトリクス記録 | 評価指標を保存 | `mlflow.log_metric()` |
# MAGIC | モデル記録 | 学習済みモデルを保存 | `mlflow.sklearn.log_model()` |
# MAGIC | タグ | メタデータを追加 | `mlflow.set_tag()` |
# MAGIC | オートロギング | 自動で記録 | `mlflow.autolog()` |
# MAGIC 
# MAGIC ### 実験管理の利点
# MAGIC 
# MAGIC 1. **再現性**: パラメータと環境を完全に記録
# MAGIC 2. **比較**: 複数の実験結果を簡単に比較
# MAGIC 3. **トレーサビリティ**: どのモデルがどのデータで学習されたか追跡
# MAGIC 4. **コラボレーション**: チーム間で実験結果を共有
# MAGIC 
# MAGIC 次のデモでは、Model Registryを使ったモデル管理を行います。
