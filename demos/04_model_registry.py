# Databricks notebook source
# MAGIC %md
# MAGIC # デモ4: モデルレジストリの操作
# MAGIC 
# MAGIC このノートブックでは、Unity Catalogのモデルレジストリを使ってモデルを管理します。
# MAGIC 
# MAGIC ## 学習目標
# MAGIC - モデルレジストリの概念を理解する
# MAGIC - モデルの登録とバージョン管理を学ぶ
# MAGIC - モデルのエイリアス(Champion/Challenger)を理解する

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 環境設定

# COMMAND ----------

import mlflow
import mlflow.sklearn

# サーバーレス環境用: レジストリURIを明示的に設定
mlflow.set_registry_uri("databricks-uc")
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# COMMAND ----------

# Unity Catalogをモデルレジストリとして使用
mlflow.set_registry_uri("databricks-uc")

# 現在のユーザー名を取得
username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')

# カタログとスキーマの設定
CATALOG = "main"
SCHEMA = f"ds_workshop_{clean_username}"

# スキーマの作成
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# モデル名の設定
MODEL_NAME = f"{CATALOG}.{SCHEMA}.wine_quality_model"

print(f"カタログ: {CATALOG}")
print(f"スキーマ: {SCHEMA}")
print(f"モデル名: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. データとモデルの準備

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
df["label"] = (df["quality"] >= 6).astype(int)

feature_cols = [c for c in df.columns if c not in ["quality", "label"]]
X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("データ準備完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. モデルの学習とレジストリへの登録

# COMMAND ----------

# エクスペリメントの設定
experiment_name = f"/Users/{username}/wine_quality_registry_demo"
mlflow.set_experiment(experiment_name)

# モデル学習とレジストリへの登録
with mlflow.start_run(run_name="model_for_registry") as run:
    # パイプライン構築と学習
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # 評価
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    mlflow.log_metric("test_auc", auc)
    mlflow.log_param("model_type", "LogisticRegression")
    
    # モデルの記録と登録
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    
    print(f"AUC: {auc:.4f}")
    print(f"モデルを {MODEL_NAME} に登録しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. モデルレジストリの確認

# COMMAND ----------

# MLflow Clientの初期化
client = MlflowClient()

# 登録されたモデルの情報を取得
try:
    model_info = client.get_registered_model(MODEL_NAME)
    print(f"モデル名: {model_info.name}")
    print(f"説明: {model_info.description or '(未設定)'}")
except Exception as e:
    print(f"モデル情報の取得エラー: {e}")

# COMMAND ----------

# モデルバージョンの一覧
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

print(f"\n登録されているバージョン数: {len(versions)}")
for v in versions:
    print(f"  バージョン {v.version}: 作成日時={v.creation_timestamp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. モデルにエイリアスを設定

# COMMAND ----------

# 最新バージョンを取得
latest_version = versions[0].version if versions else "1"

# エイリアスの設定(Champion = 本番用)
try:
    client.set_registered_model_alias(MODEL_NAME, "Champion", latest_version)
    print(f"バージョン {latest_version} に 'Champion' エイリアスを設定しました")
except Exception as e:
    print(f"エイリアス設定エラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 改良版モデルの作成と登録

# COMMAND ----------

# 異なるパラメータで新しいモデルを学習
with mlflow.start_run(run_name="improved_model") as run:
    # より強い正則化のモデル
    pipeline_v2 = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.5,
            max_iter=200,
            solver="lbfgs",
            random_state=42
        ))
    ])
    
    pipeline_v2.fit(X_train, y_train)
    
    # 評価
    y_pred_proba_v2 = pipeline_v2.predict_proba(X_test)[:, 1]
    auc_v2 = roc_auc_score(y_test, y_pred_proba_v2)
    
    mlflow.log_metric("test_auc", auc_v2)
    mlflow.log_param("model_type", "LogisticRegression_v2")
    
    # 新しいバージョンとして登録
    mlflow.sklearn.log_model(
        pipeline_v2,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    
    print(f"改良版モデル AUC: {auc_v2:.4f}")
    print("新しいバージョンを登録しました")

# COMMAND ----------

# 新しいバージョンをChallengerとして設定
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = versions[0].version

try:
    client.set_registered_model_alias(MODEL_NAME, "Challenger", latest_version)
    print(f"バージョン {latest_version} に 'Challenger' エイリアスを設定しました")
except Exception as e:
    print(f"エイリアス設定エラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. エイリアスによるモデル読み込み

# COMMAND ----------

# Championモデルの読み込み
champion_model_uri = f"models:/{MODEL_NAME}@Champion"
print(f"Champion URI: {champion_model_uri}")

try:
    champion_model = mlflow.sklearn.load_model(champion_model_uri)
    print("Championモデルを読み込みました")
    
    # 予測実行
    sample_data = X_test.iloc[:5]
    champion_predictions = champion_model.predict(sample_data)
    print(f"予測結果: {champion_predictions}")
except Exception as e:
    print(f"読み込みエラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. モデルの説明の更新

# COMMAND ----------

# モデルの説明を更新
try:
    client.update_registered_model(
        name=MODEL_NAME,
        description="Wine Quality分類モデル: ワインの品質を予測するロジスティック回帰モデル"
    )
    print("モデルの説明を更新しました")
except Exception as e:
    print(f"更新エラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 
# MAGIC このデモで学んだ内容:
# MAGIC 
# MAGIC ### Model Registryの主要機能
# MAGIC 
# MAGIC | 機能 | 説明 |
# MAGIC |------|------|
# MAGIC | モデル登録 | `registered_model_name`パラメータで自動登録 |
# MAGIC | バージョン管理 | 登録するたびに新しいバージョンが作成される |
# MAGIC | エイリアス | Champion/Challengerなどの役割を割り当て |
# MAGIC | 説明 | モデルの用途やメタデータを記録 |
# MAGIC 
# MAGIC ### エイリアスの活用
# MAGIC 
# MAGIC - **Champion**: 本番環境で使用するモデル
# MAGIC - **Challenger**: 次の候補となるモデル(A/Bテスト用)
# MAGIC - バージョン番号ではなくエイリアスで参照することで、コード変更なしにモデルを切り替え可能
# MAGIC 
# MAGIC ### Unity Catalog統合の利点
# MAGIC 
# MAGIC 1. **ガバナンス**: データと同じ権限モデルでモデルを管理
# MAGIC 2. **リネージ**: モデルがどのデータで学習されたか追跡
# MAGIC 3. **発見性**: カタログブラウザでモデルを検索可能
# MAGIC 4. **セキュリティ**: きめ細かいアクセス制御
# MAGIC 
# MAGIC 次のデモでは、バッチ推論を行います。
