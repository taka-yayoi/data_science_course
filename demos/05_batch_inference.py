# Databricks notebook source
# MAGIC %md
# MAGIC # デモ5: バッチ推論
# MAGIC 
# MAGIC このノートブックでは、登録済みモデルを使ってバッチ推論を行います。
# MAGIC 
# MAGIC ## 学習目標
# MAGIC - Model Registryからモデルを読み込む方法を学ぶ
# MAGIC - バッチ推論のワークフローを体験する
# MAGIC - 推論結果をDelta Tableに保存する

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
from pyspark.sql.functions import current_timestamp, lit, pandas_udf
from pyspark.sql.types import DoubleType

# Unity Catalogをモデルレジストリとして使用
mlflow.set_registry_uri("databricks-uc")

# ユーザー情報
username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')

# カタログとスキーマ
CATALOG = "workspace"
SCHEMA = f"ds_workshop_{clean_username}"

# スキーマの作成(存在しない場合)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# モデル名
MODEL_NAME = f"{CATALOG}.{SCHEMA}.wine_quality_model"

print(f"カタログ: {CATALOG}")
print(f"スキーマ: {SCHEMA}")
print(f"モデル: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 推論用データの準備

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# Wine Qualityデータセットの読み込み
wine_df = spark.read.csv(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

# 推論用データ(ラベルなし)を作成
# 実際の運用では、新しいデータがここに入る
inference_df = wine_df.drop("quality").withColumn("wine_id", monotonically_increasing_id())

print(f"推論対象レコード数: {inference_df.count()}")
display(inference_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. モデルの読み込み

# COMMAND ----------

# Championモデルを読み込み(エイリアス使用)
try:
    model_uri = f"models:/{MODEL_NAME}@Champion"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"モデルを読み込みました: {model_uri}")
except Exception as e:
    # Championがない場合は最新バージョンを使用
    print(f"Championモデルが見つかりません。最新バージョンを使用します。")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest_version = versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{latest_version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"モデルを読み込みました: {model_uri}")
    else:
        raise Exception(f"モデル {MODEL_NAME} が見つかりません。先にデモ4を実行してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. バッチ推論の実行

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法1: pandasに変換して推論(小規模データ向け)

# COMMAND ----------

# pandasに変換
inference_pdf = inference_df.toPandas()

# 特徴量カラムを取得(wine_idを除く)
feature_cols = [c for c in inference_pdf.columns if c != "wine_id"]

# 推論の実行
predictions = model.predict(inference_pdf[feature_cols])
probabilities = model.predict_proba(inference_pdf[feature_cols])[:, 1]

# 結果をDataFrameに追加
inference_pdf["prediction"] = predictions
inference_pdf["probability"] = probabilities

print("予測結果のサンプル:")
inference_pdf[["wine_id", "prediction", "probability"]].head(10)

# COMMAND ----------

# 推論結果の統計
print("予測結果の分布:")
print(inference_pdf["prediction"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法2: Spark UDFを使った推論(大規模データ向け)

# COMMAND ----------

# pandas UDFで推論関数を定義
# 注意: Free Editionのサーバーレスでは制限がある場合があります

# @pandas_udf(DoubleType())
# def predict_udf(*cols):
#     X = pd.concat(cols, axis=1)
#     X.columns = feature_cols
#     return pd.Series(model.predict_proba(X)[:, 1])

# 上記UDFが使えない場合は方法1を使用

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 推論結果をDelta Tableに保存

# COMMAND ----------

# Spark DataFrameに変換
results_spark_df = spark.createDataFrame(inference_pdf[["wine_id", "prediction", "probability"]])

# タイムスタンプとモデルバージョンを追加
results_df = results_spark_df \
    .withColumn("inference_timestamp", current_timestamp()) \
    .withColumn("model_version", lit(model_uri))

display(results_df.limit(5))

# COMMAND ----------

# Delta Tableとして保存
output_table = f"{CATALOG}.{SCHEMA}.wine_predictions"

results_df.write.format("delta").mode("overwrite").saveAsTable(output_table)

print(f"推論結果を {output_table} に保存しました")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 保存された結果を確認
# MAGIC SELECT * FROM wine_predictions LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 予測結果の集計
# MAGIC SELECT 
# MAGIC   CASE 
# MAGIC     WHEN prediction = 1 THEN '高品質' 
# MAGIC     ELSE '標準品質' 
# MAGIC   END as quality_label,
# MAGIC   COUNT(*) as count,
# MAGIC   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
# MAGIC FROM wine_predictions
# MAGIC GROUP BY prediction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 推論パイプラインの自動化(参考)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ジョブとしてスケジュール実行する場合
# MAGIC 
# MAGIC 1. このノートブックをジョブとして登録
# MAGIC 2. スケジュール(毎日、毎週など)を設定
# MAGIC 3. 新しいデータが到着するたびに自動で推論を実行
# MAGIC 
# MAGIC ```python
# MAGIC # 実際の運用では、新しいデータソースを指定
# MAGIC new_data = spark.read.format("delta").table("new_wine_samples")
# MAGIC # ... 推論処理 ...
# MAGIC results_df.write.format("delta").mode("append").saveAsTable(output_table)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. クリーンアップ(オプション)

# COMMAND ----------

# テーブルの削除(必要に応じてコメントアウトを解除)
# spark.sql(f"DROP TABLE IF EXISTS {output_table}")
# print(f"テーブル {output_table} を削除しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 
# MAGIC このデモで学んだ内容:
# MAGIC 
# MAGIC ### バッチ推論のワークフロー
# MAGIC 
# MAGIC 1. **モデル読み込み**: エイリアス(`@Champion`)またはバージョン番号でモデルを指定
# MAGIC 2. **推論実行**: scikit-learnモデルで予測
# MAGIC 3. **結果保存**: Delta Tableに推論結果を保存(追記または上書き)
# MAGIC 
# MAGIC ### 運用上のポイント
# MAGIC 
# MAGIC | ポイント | 説明 |
# MAGIC |----------|------|
# MAGIC | エイリアス使用 | バージョン番号ではなくエイリアスで参照することで、コード変更なしにモデル更新が可能 |
# MAGIC | メタデータ付与 | 推論タイムスタンプとモデルバージョンを記録して追跡性を確保 |
# MAGIC | Delta Lake | ACID特性とバージョン管理により、安全なデータ管理 |
# MAGIC | ジョブスケジュール | 定期的なバッチ推論を自動化 |
# MAGIC 
# MAGIC ### MLOpsとの関連
# MAGIC 
# MAGIC - **這う**: ノートブックで手動実行
# MAGIC - **歩く**: ジョブとしてスケジュール実行
# MAGIC - **走る**: CI/CDパイプラインと統合、モニタリングを追加
