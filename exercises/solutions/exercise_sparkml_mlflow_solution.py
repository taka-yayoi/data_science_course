# Databricks notebook source
# MAGIC %md
# MAGIC # 実習: SparkML + MLflowによる機械学習ワークフロー【解答】
# MAGIC 
# MAGIC この実習では、Wine Qualityデータセットを使って機械学習モデルを構築し、
# MAGIC MLflowで実験を管理します。
# MAGIC 
# MAGIC ## 実習の流れ
# MAGIC 1. データの準備と探索
# MAGIC 2. SparkML Pipelineの構築
# MAGIC 3. モデルの学習と評価
# MAGIC 4. MLflowによる実験記録
# MAGIC 5. ハイパーパラメータチューニング
# MAGIC 6. ベストモデルの登録

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 環境設定

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 実験名の設定
username = spark.sql("SELECT current_user()").collect()[0][0]
experiment_name = f"/Users/{username}/wine_quality_exercise"
mlflow.set_experiment(experiment_name)

print(f"実験名: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. データの準備

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 データの読み込み

# COMMAND ----------

# Wine Qualityデータセットの読み込み
wine_df = spark.read.csv(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

# カラム名のクリーンアップ(スペースをアンダースコアに)
for col_name in wine_df.columns:
    wine_df = wine_df.withColumnRenamed(col_name, col_name.replace(" ", "_"))

print(f"レコード数: {wine_df.count()}")
print(f"カラム: {wine_df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 データの確認

# COMMAND ----------

# データの確認
display(wine_df.limit(10))

# COMMAND ----------

# 【解答】統計サマリーを表示
display(wine_df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 ラベルの作成

# COMMAND ----------

# 【解答】ラベルカラムを追加
wine_df = wine_df.withColumn(
    "label",
    when(col("quality") >= 6, 1.0).otherwise(0.0)
)

# ラベルの分布を確認
display(wine_df.groupBy("label").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 特徴量の定義

# COMMAND ----------

# 特徴量カラムのリスト(qualityとlabel以外)
feature_cols = [c for c in wine_df.columns if c not in ["quality", "label"]]
print(f"特徴量: {feature_cols}")
print(f"特徴量数: {len(feature_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 データの分割

# COMMAND ----------

# 【解答】データを訓練用(80%)とテスト用(20%)に分割
train_df, test_df = wine_df.randomSplit([0.8, 0.2], seed=42)

print(f"訓練データ: {train_df.count()} 件")
print(f"テストデータ: {test_df.count()} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. SparkML Pipelineの構築

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 コンポーネントの定義

# COMMAND ----------

# 【解答】VectorAssemblerを作成
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features"
)

# COMMAND ----------

# 【解答】StandardScalerを作成
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

# COMMAND ----------

# 【解答】LogisticRegressionを作成
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Pipelineの構築

# COMMAND ----------

# 【解答】Pipelineを作成
pipeline = Pipeline(stages=[assembler, scaler, lr])

print("Pipeline構成:")
for i, stage in enumerate(pipeline.getStages()):
    print(f"  {i+1}. {type(stage).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. モデルの学習と評価

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 モデルの学習

# COMMAND ----------

# 【解答】Pipelineを訓練データで学習
model = pipeline.fit(train_df)

print("モデルの学習が完了しました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 予測の実行

# COMMAND ----------

# 【解答】テストデータで予測を実行
predictions = model.transform(test_df)

# 予測結果の確認
display(predictions.select("label", "prediction", "probability").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 モデルの評価

# COMMAND ----------

# 評価器の作成
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

# 【解答】各評価指標を計算
auc = auc_evaluator.evaluate(predictions)
accuracy = acc_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

print("=" * 40)
print("モデル評価結果")
print("=" * 40)
print(f"AUC-ROC:  {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MLflowによる実験記録

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 手動ロギング

# COMMAND ----------

# MLflowで実験を記録
with mlflow.start_run(run_name="exercise_baseline"):
    
    # 【解答】パラメータを記録
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("regParam", 0.01)
    mlflow.log_param("algorithm", "LogisticRegression")
    
    # 【解答】メトリクスを記録
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # 【解答】モデルを記録
    mlflow.spark.log_model(model, "spark_model")
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print("実験を記録しました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ハイパーパラメータチューニング

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 複数のパラメータで実験

# COMMAND ----------

# 試すパラメータの組み合わせ
param_grid = [
    {"maxIter": 50, "regParam": 0.001},
    {"maxIter": 100, "regParam": 0.01},
    {"maxIter": 100, "regParam": 0.1},
    {"maxIter": 200, "regParam": 0.01},
]

results = []

for i, params in enumerate(param_grid):
    with mlflow.start_run(run_name=f"experiment_{i+1}"):
        # パラメータの記録
        mlflow.log_params(params)
        
        # モデル構築
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=params["maxIter"],
            regParam=params["regParam"]
        )
        pipeline = Pipeline(stages=[assembler, scaler, lr])
        
        # 学習と評価
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        
        auc = auc_evaluator.evaluate(predictions)
        accuracy = acc_evaluator.evaluate(predictions)
        
        # メトリクスの記録
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_accuracy", accuracy)
        
        results.append({
            "params": params,
            "auc": auc,
            "accuracy": accuracy,
            "run_id": mlflow.active_run().info.run_id
        })
        
        print(f"実験 {i+1}: maxIter={params['maxIter']}, regParam={params['regParam']} -> AUC={auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 ベストモデルの特定

# COMMAND ----------

# 【解答】AUCが最も高い結果を見つける
best_result = max(results, key=lambda x: x["auc"])

print("=" * 50)
print("ベストモデル")
print("=" * 50)
print(f"パラメータ: {best_result['params']}")
print(f"AUC: {best_result['auc']:.4f}")
print(f"Run ID: {best_result['run_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ベストモデルの登録

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Unity Catalogにモデルを登録

# COMMAND ----------

# Unity Catalogの設定
mlflow.set_registry_uri("databricks-uc")

clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')
CATALOG = "main"
SCHEMA = f"ds_workshop_{clean_username}"

# スキーマの作成
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

MODEL_NAME = f"{CATALOG}.{SCHEMA}.wine_quality_exercise_model"

print(f"モデル名: {MODEL_NAME}")

# COMMAND ----------

# 【解答】ベストモデルを登録
best_run_id = best_result["run_id"]

try:
    result = mlflow.register_model(
        f"runs:/{best_run_id}/spark_model",
        MODEL_NAME
    )
    print(f"モデルを登録しました: {result.name} (バージョン: {result.version})")
except Exception as e:
    print(f"登録エラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. まとめと振り返り

# COMMAND ----------

# MAGIC %md
# MAGIC ### 実習で学んだこと
# MAGIC 
# MAGIC #### SparkML Pipeline
# MAGIC - **VectorAssembler**: 複数の特徴量を1つのベクトルに結合
# MAGIC - **StandardScaler**: 特徴量の標準化
# MAGIC - **LogisticRegression**: 二値分類モデル
# MAGIC - **Pipeline**: 複数のステージを連結
# MAGIC 
# MAGIC #### MLflow Tracking
# MAGIC - `mlflow.log_param()`: パラメータの記録
# MAGIC - `mlflow.log_metric()`: メトリクスの記録
# MAGIC - `mlflow.spark.log_model()`: モデルの記録
# MAGIC 
# MAGIC #### Model Registry
# MAGIC - `mlflow.register_model()`: モデルの登録

# COMMAND ----------

# MAGIC %md
# MAGIC ## おまけ: RandomForestで試してみよう

# COMMAND ----------

# RandomForestClassifierでモデルを構築
from pyspark.ml.classification import RandomForestClassifier

with mlflow.start_run(run_name="random_forest_challenge"):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=5
    )
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    
    auc = auc_evaluator.evaluate(predictions)
    accuracy = acc_evaluator.evaluate(predictions)
    
    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 5)
    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.spark.log_model(model, "spark_model")
    
    print(f"RandomForest AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
