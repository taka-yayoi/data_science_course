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
from mlflow.tracking import MlflowClient

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

print(f"モデル名: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. モデルの準備(前のデモで作成したモデルを使用)

# COMMAND ----------

from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# データの準備
wine_df = spark.read.csv(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

for col_name in wine_df.columns:
    wine_df = wine_df.withColumnRenamed(col_name, col_name.replace(" ", "_"))

wine_df = wine_df.withColumn(
    "label",
    when(col("quality") >= 6, 1.0).otherwise(0.0)
)

feature_cols = [c for c in wine_df.columns if c not in ["quality", "label"]]
train_df, test_df = wine_df.randomSplit([0.8, 0.2], seed=42)

print("データ準備完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. モデルの学習とレジストリへの登録

# COMMAND ----------

# 実験の設定
experiment_name = f"/Users/{username}/wine_quality_registry_demo"
mlflow.set_experiment(experiment_name)

# モデル学習とレジストリへの登録
with mlflow.start_run(run_name="model_for_registry") as run:
    # Pipeline構築と学習
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01)
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(train_df)
    
    # 評価
    predictions = model.transform(test_df)
    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(predictions)
    
    mlflow.log_metric("test_auc", auc)
    
    # モデルの記録と登録
    mlflow.spark.log_model(
        model,
        artifact_path="spark_model",
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
    print(f"作成日時: {model_info.creation_timestamp}")
except Exception as e:
    print(f"モデル情報の取得エラー: {e}")

# COMMAND ----------

# モデルバージョンの一覧
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

print(f"\n登録されているバージョン数: {len(versions)}")
for v in versions:
    print(f"  バージョン {v.version}: 状態={v.status}, 作成日時={v.creation_timestamp}")

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
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=200,
        regParam=0.001,
        elasticNetParam=0.5
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model_v2 = pipeline.fit(train_df)
    
    # 評価
    predictions_v2 = model_v2.transform(test_df)
    auc_v2 = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(predictions_v2)
    
    mlflow.log_metric("test_auc", auc_v2)
    mlflow.log_param("model_type", "improved")
    
    # 新しいバージョンとして登録
    mlflow.spark.log_model(
        model_v2,
        artifact_path="spark_model",
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
    champion_model = mlflow.spark.load_model(champion_model_uri)
    print("Championモデルを読み込みました")
    
    # 予測実行
    champion_predictions = champion_model.transform(test_df.limit(5))
    display(champion_predictions.select("label", "prediction"))
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
# MAGIC ## 9. Unity Catalogでのモデル確認

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Unity Catalogでモデルを確認
# MAGIC SHOW MODELS IN main.${SCHEMA}

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
