# Databricks notebook source
# MAGIC %md
# MAGIC # デモ4: Unity Catalogモデルレジストリ
# MAGIC
# MAGIC このノートブックでは、MLflowモデルをUnity Catalogに登録し、バージョン管理を行います。
# MAGIC
# MAGIC **重要:** サーバレスv2を使用している場合は、以下のセルでMLflowをアップグレードしてください。
# MAGIC サーバレスv4の場合はスキップできます。
# MAGIC
# MAGIC 参考: [Databricks Free EditionでUnity Catalogモデルレジストリがエラーになる場合の対処法](https://qiita.com/taka_yayoi/items/6068b9bb4eb05ab5ddbd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (オプション) MLflowアップグレード
# MAGIC サーバレスv2でUCモデルレジストリを使う場合に必要です。

# COMMAND ----------

# サーバレスv2の場合は以下のコメントを外して実行
# %pip install --upgrade mlflow -q

# COMMAND ----------

# 上記を実行した場合は、このセルも実行してPython環境を再起動
# dbutils.library.restartPython()

# COMMAND ----------

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlflow.models import infer_signature

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. カタログとスキーマの設定

# COMMAND ----------

# ユーザー名を取得してカタログ名に使用
username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')

# カタログとスキーマの設定
CATALOG = f"ds_workshop_{clean_username}"
SCHEMA = "ml"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.wine_quality_logreg"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.wine_predictions"

print(f"カタログ: {CATALOG}")
print(f"スキーマ: {SCHEMA}")
print(f"モデル名: {MODEL_NAME}")

# COMMAND ----------

# カタログとスキーマの作成
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. データの準備

# COMMAND ----------

# UCI Wine Qualityデータセットを読み込み
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# 二値分類に変換(クラス0 vs その他)
df["target_binary"] = (df["target"] == 0).astype(int)

print(f"データ形状: {df.shape}")
print(f"ターゲット分布:\n{df['target_binary'].value_counts()}")

# COMMAND ----------

# 特徴量とターゲットの分離
feature_cols = wine.feature_names
X = df[feature_cols]
y = df["target_binary"]

# 訓練/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. モデル学習とMLflowログ

# COMMAND ----------

# パイプラインの構築
classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])

# COMMAND ----------

# MLflow設定
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# モデル学習とログ
with mlflow.start_run(run_name="UC-Wine-LogReg") as run:
    # 学習
    classifier.fit(X_train, y_train)
    
    # 予測と評価
    pred = classifier.predict(X_test)
    pred_proba = classifier.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    auc = roc_auc_score(y_test, pred_proba)
    
    # パラメータとメトリクスをログ
    mlflow.log_params({"model": "LogisticRegression", "class_weight": "balanced"})
    mlflow.log_metrics({"ACC": acc, "F1_weighted": f1, "AUC": auc})
    
    # シグネチャの推論
    sig = infer_signature(X_train, classifier.predict(X_train))
    
    # モデルをログ
    mlflow.sklearn.log_model(
        sk_model=classifier,
        artifact_path="model",
        signature=sig,
        input_example=X_train.head(2)
    )
    
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"ACC: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. モデルレジストリへの登録

# COMMAND ----------

# モデルをUnity Catalogに登録
model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

time.sleep(3)

# Championエイリアスを設定
client.set_registered_model_alias(name=MODEL_NAME, alias="Champion", version=model_version.version)

print(f"✅ Registered to UC: {MODEL_NAME} v{model_version.version} (alias=Champion)")
print(f"   ACC={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 登録モデルの確認

# COMMAND ----------

# 登録されたモデルの情報を表示
model_info = client.get_registered_model(MODEL_NAME)
print(f"モデル名: {model_info.name}")
print(f"作成日時: {model_info.creation_timestamp}")

# エイリアス一覧(Databricks UCでは辞書型で返る)
if model_info.aliases:
    print(f"\nエイリアス:")
    for alias, version in model_info.aliases.items():
        print(f"  - @{alias} -> Version {version}")
else:
    print("\nエイリアス: なし")

# バージョン一覧
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
print(f"\nバージョン数: {len(versions)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 改良版モデルの作成と登録

# COMMAND ----------

# より強い正則化のモデル
classifier_v2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.5,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    ))
])

# 学習とログ
with mlflow.start_run(run_name="UC-Wine-LogReg-v2") as run_v2:
    classifier_v2.fit(X_train, y_train)
    
    pred_v2 = classifier_v2.predict(X_test)
    pred_proba_v2 = classifier_v2.predict_proba(X_test)[:, 1]
    
    acc_v2 = accuracy_score(y_test, pred_v2)
    f1_v2 = f1_score(y_test, pred_v2, average="weighted")
    auc_v2 = roc_auc_score(y_test, pred_proba_v2)
    
    mlflow.log_params({"model": "LogisticRegression_v2", "C": 0.5, "class_weight": "balanced"})
    mlflow.log_metrics({"ACC": acc_v2, "F1_weighted": f1_v2, "AUC": auc_v2})
    
    sig_v2 = infer_signature(X_train, classifier_v2.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model=classifier_v2,
        artifact_path="model",
        signature=sig_v2,
        input_example=X_train.head(2)
    )
    
    run_id_v2 = run_v2.info.run_id
    print(f"Run ID: {run_id_v2}")
    print(f"ACC: {acc_v2:.4f}, F1: {f1_v2:.4f}, AUC: {auc_v2:.4f}")

# COMMAND ----------

# 新しいバージョンを登録
model_uri_v2 = f"runs:/{run_id_v2}/model"
model_version_v2 = mlflow.register_model(model_uri=model_uri_v2, name=MODEL_NAME)

time.sleep(3)

# Challengerエイリアスを設定
client.set_registered_model_alias(name=MODEL_NAME, alias="Challenger", version=model_version_v2.version)

print(f"✅ Registered to UC: {MODEL_NAME} v{model_version_v2.version} (alias=Challenger)")
print(f"   ACC={acc_v2:.3f}, F1={f1_v2:.3f}, AUC={auc_v2:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. モデルの読み込みと推論

# COMMAND ----------

# Championモデルを読み込み
loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Champion")

# 全データで予測
pred_all = loaded_model.predict(X)
pred_proba_all = loaded_model.predict_proba(X)[:, 1]

print(f"予測完了: {len(pred_all)}件")

# COMMAND ----------

# 予測結果をDataFrameに変換
pred_df = pd.DataFrame({
    "sample_id": np.arange(len(df)),
    "prediction": pred_all.astype(int),
    "probability": pred_proba_all.astype(float),
    "actual": y.values
})

# Sparkテーブルとして保存
pred_sdf = spark.createDataFrame(pred_df)
pred_sdf.write.mode("overwrite").saveAsTable(PRED_TABLE)

display(spark.table(PRED_TABLE))
print(f"✅ 推論テーブルを作成/更新: {PRED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. まとめ
# MAGIC
# MAGIC ### 学んだこと
# MAGIC
# MAGIC | 操作 | コード |
# MAGIC |------|--------|
# MAGIC | レジストリ設定 | `mlflow.set_registry_uri("databricks-uc")` |
# MAGIC | モデル登録 | `mlflow.register_model(model_uri, name)` |
# MAGIC | エイリアス設定 | `client.set_registered_model_alias(name, alias, version)` |
# MAGIC | モデル読み込み | `mlflow.sklearn.load_model("models:/name@alias")` |
# MAGIC
# MAGIC ### Champion/Challengerパターン
# MAGIC - **Champion**: 本番運用中のモデル
# MAGIC - **Challenger**: 評価中の新モデル
# MAGIC - エイリアスを切り替えるだけでモデルを入れ替え可能

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ(必要に応じて実行)

# COMMAND ----------

# # モデルとテーブルの削除
# client.delete_registered_model(MODEL_NAME)
# spark.sql(f"DROP TABLE IF EXISTS {PRED_TABLE}")
# spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
# spark.sql(f"DROP CATALOG IF EXISTS {CATALOG} CASCADE")
# print("クリーンアップ完了")
