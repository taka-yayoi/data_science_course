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
import mlflow.spark
from mlflow.models.signature import infer_signature

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

from pyspark.sql.functions import when, col

# Wine Qualityデータセットの読み込み
wine_df = spark.read.csv(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

# カラム名のクリーンアップ
for col_name in wine_df.columns:
    wine_df = wine_df.withColumnRenamed(col_name, col_name.replace(" ", "_"))

# 二値分類用のラベル作成
wine_df = wine_df.withColumn(
    "label",
    when(col("quality") >= 6, 1.0).otherwise(0.0)
)

# 特徴量カラム
feature_cols = [c for c in wine_df.columns if c not in ["quality", "label"]]

# データ分割
train_df, test_df = wine_df.randomSplit([0.8, 0.2], seed=42)

print(f"訓練データ: {train_df.count()} 件")
print(f"テストデータ: {test_df.count()} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. MLflowによる実験記録

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 手動ロギング

# COMMAND ----------

# ハイパーパラメータの設定
params = {
    "maxIter": 100,
    "regParam": 0.01,
    "elasticNetParam": 0.5
}

# MLflowラン開始
with mlflow.start_run(run_name="logistic_regression_manual"):
    
    # パラメータの記録
    mlflow.log_params(params)
    mlflow.log_param("features", feature_cols)
    
    # Pipeline構築
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=params["maxIter"],
        regParam=params["regParam"],
        elasticNetParam=params["elasticNetParam"]
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    
    # モデル学習
    model = pipeline.fit(train_df)
    
    # 予測
    predictions = model.transform(test_df)
    
    # 評価指標の計算
    auc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    
    auc = auc_evaluator.evaluate(predictions)
    accuracy = acc_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    
    # メトリクスの記録
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # モデルの記録
    mlflow.spark.log_model(model, "spark_model")
    
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
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
    {"maxIter": 50, "regParam": 0.001},
    {"maxIter": 100, "regParam": 0.1},
    {"maxIter": 200, "regParam": 0.01},
]

for i, params in enumerate(params_list):
    with mlflow.start_run(run_name=f"autolog_experiment_{i+1}"):
        # タグの追加
        mlflow.set_tag("experiment_type", "hyperparameter_search")
        mlflow.set_tag("iteration", i+1)
        
        # Pipeline構築
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=params["maxIter"],
            regParam=params["regParam"]
        )
        
        pipeline = Pipeline(stages=[assembler, scaler, lr])
        
        # モデル学習(autologにより自動記録)
        model = pipeline.fit(train_df)
        
        # 予測と評価
        predictions = model.transform(test_df)
        auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(predictions)
        
        # 追加メトリクスの記録
        mlflow.log_metric("test_auc", auc)
        
        print(f"実験 {i+1}: maxIter={params['maxIter']}, regParam={params['regParam']} -> AUC={auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 実験結果の確認

# COMMAND ----------

# 実験のランを取得
experiment = mlflow.get_experiment_by_name(experiment_name)
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# 結果の表示
display(runs_df[["run_id", "metrics.test_auc", "params.maxIter", "params.regParam", "tags.mlflow.runName"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ベストモデルの特定

# COMMAND ----------

# AUCが最も高いランを取得
best_run = runs_df.loc[runs_df["metrics.test_auc"].idxmax()]

print("=" * 50)
print("ベストモデル")
print("=" * 50)
print(f"Run ID: {best_run['run_id']}")
print(f"Run Name: {best_run['tags.mlflow.runName']}")
print(f"Test AUC: {best_run['metrics.test_auc']:.4f}")
print(f"Parameters:")
print(f"  maxIter: {best_run.get('params.maxIter', 'N/A')}")
print(f"  regParam: {best_run.get('params.regParam', 'N/A')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. モデルの読み込みと推論

# COMMAND ----------

# ベストモデルの読み込み
best_run_id = best_run["run_id"]
loaded_model = mlflow.spark.load_model(f"runs:/{best_run_id}/spark_model")

# 新しいデータで予測
sample_data = test_df.limit(5)
new_predictions = loaded_model.transform(sample_data)
display(new_predictions.select("label", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MLflow UIへのリンク

# COMMAND ----------

# 実験へのリンクを表示
experiment_url = f"#mlflow/experiments/{experiment.experiment_id}"
print(f"MLflow実験ページ: {experiment_url}")
print("\n左サイドバーの「実験」から確認することもできます。")

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
# MAGIC | モデル記録 | 学習済みモデルを保存 | `mlflow.spark.log_model()` |
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
