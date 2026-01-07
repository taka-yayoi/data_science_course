# Databricks notebook source
# MAGIC %md
# MAGIC # デモ5: バッチ推論
# MAGIC
# MAGIC このノートブックでは、Unity Catalogに登録したモデルを使ってバッチ推論を行います。
# MAGIC
# MAGIC **前提:** デモ4を実行してモデルが登録されていること

# COMMAND ----------

import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 設定

# COMMAND ----------

# ユーザー名を取得
username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')

# カタログとスキーマ(04_model_registryと同じ)
CATALOG = f"ds_workshop_{clean_username}"
SCHEMA = "ml"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.wine_quality_logreg"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.wine_batch_predictions"

# MLflow設定
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

print(f"カタログ: {CATALOG}")
print(f"スキーマ: {SCHEMA}")
print(f"モデル: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 推論用データの準備

# COMMAND ----------

# Wine Qualityデータセットを読み込み(新しいデータとして扱う)
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
df["target_binary"] = (df["target"] == 0).astype(int)

# 特徴量
feature_cols = wine.feature_names
X = df[feature_cols]
y = df["target_binary"]

print(f"推論対象データ: {X.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Championモデルの読み込み

# COMMAND ----------

# Championモデルを読み込み
try:
    model_uri = f"models:/{MODEL_NAME}@Champion"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ モデルを読み込みました: {model_uri}")
except Exception as e:
    print(f"❌ Championモデルが見つかりません: {e}")
    print("先にデモ4を実行してください。")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. バッチ推論の実行

# COMMAND ----------

# 推論実行
start_time = time.time()

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

elapsed_time = time.time() - start_time
print(f"推論完了: {len(predictions)}件 ({elapsed_time:.2f}秒)")

# COMMAND ----------

# 推論結果をDataFrameに変換
results_df = pd.DataFrame({
    "sample_id": np.arange(len(df)),
    "prediction": predictions.astype(int),
    "probability": probabilities.astype(float),
    "actual": y.values
})

# 正解/不正解を追加
results_df["correct"] = (results_df["prediction"] == results_df["actual"]).astype(int)

display(spark.createDataFrame(results_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 推論結果の評価

# COMMAND ----------

# 精度評価
accuracy = accuracy_score(y, predictions)
print(f"全体精度: {accuracy:.4f}")
print(f"\n分類レポート:")
print(classification_report(y, predictions, target_names=["Class 1/2", "Class 0"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 結果をDeltaテーブルに保存

# COMMAND ----------

# Sparkテーブルとして保存
results_sdf = spark.createDataFrame(results_df)
results_sdf.write.format("delta").mode("overwrite").saveAsTable(PRED_TABLE)

print(f"✅ 推論結果を保存しました: {PRED_TABLE}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 保存されたデータを確認
# MAGIC SELECT * FROM ${PRED_TABLE} LIMIT 10

# COMMAND ----------

# 保存データの確認
display(spark.sql(f"SELECT * FROM {PRED_TABLE} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 集計分析

# COMMAND ----------

# 予測分布の確認
summary = spark.sql(f"""
SELECT 
    prediction,
    COUNT(*) as count,
    AVG(probability) as avg_probability,
    SUM(correct) as correct_count
FROM {PRED_TABLE}
GROUP BY prediction
ORDER BY prediction
""")

display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Champion vs Challengerの比較(オプション)

# COMMAND ----------

# Challengerモデルがあれば比較
try:
    challenger_uri = f"models:/{MODEL_NAME}@Challenger"
    challenger_model = mlflow.sklearn.load_model(challenger_uri)
    print(f"✅ Challengerモデルを読み込みました")
    
    # Challengerで推論
    challenger_preds = challenger_model.predict(X)
    challenger_accuracy = accuracy_score(y, challenger_preds)
    
    print(f"\n=== モデル比較 ===")
    print(f"Champion精度:   {accuracy:.4f}")
    print(f"Challenger精度: {challenger_accuracy:.4f}")
    
    if challenger_accuracy > accuracy:
        print("→ Challengerの方が高精度です")
    else:
        print("→ Championの方が高精度です")
        
except Exception as e:
    print(f"Challengerモデルがありません: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. まとめ
# MAGIC
# MAGIC ### バッチ推論のワークフロー
# MAGIC 1. Unity Catalogからモデルを読み込み
# MAGIC 2. 推論対象データを準備
# MAGIC 3. `model.predict()`でバッチ推論
# MAGIC 4. 結果をDeltaテーブルに保存
# MAGIC 5. 後続の分析やレポーティングに活用
# MAGIC
# MAGIC ### 本番運用での考慮点
# MAGIC - ジョブとしてスケジュール実行
# MAGIC - データの増分処理(新規データのみ推論)
# MAGIC - モニタリングとアラート設定

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ(必要に応じて実行)

# COMMAND ----------

# # テーブルの削除
# spark.sql(f"DROP TABLE IF EXISTS {PRED_TABLE}")
# print(f"テーブル {PRED_TABLE} を削除しました")
