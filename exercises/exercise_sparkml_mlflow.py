# Databricks notebook source
# MAGIC %md
# MAGIC # 実習: SparkML + MLflowによる機械学習ワークフロー
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
# MAGIC 
# MAGIC ## 所要時間: 約60-90分
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ### 穴埋め問題について
# MAGIC - `# TODO: ` のコメントがある箇所を埋めてください
# MAGIC - `___` の部分に適切なコードを記入してください
# MAGIC - 分からない場合はヒントを参照してください

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

# TODO: 統計サマリーを表示してください
# ヒント: display(wine_df.___())
display(wine_df.___)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 ラベルの作成
# MAGIC 
# MAGIC qualityカラム(3-8の整数)を二値分類問題に変換します。
# MAGIC - quality >= 6 → 高品質 (label = 1.0)
# MAGIC - quality < 6 → 標準品質 (label = 0.0)

# COMMAND ----------

# TODO: ラベルカラムを追加してください
# ヒント: when(col("quality") >= 6, 1.0).otherwise(0.0)
wine_df = wine_df.withColumn(
    "label",
    ___
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

# TODO: データを訓練用(80%)とテスト用(20%)に分割してください
# ヒント: randomSplit([訓練割合, テスト割合], seed=42)
train_df, test_df = wine_df.___

print(f"訓練データ: {train_df.count()} 件")
print(f"テストデータ: {test_df.count()} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. SparkML Pipelineの構築

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 コンポーネントの定義

# COMMAND ----------

# TODO: VectorAssemblerを作成してください
# - inputCols: 特徴量カラムのリスト
# - outputCol: "raw_features"
assembler = VectorAssembler(
    inputCols=___,
    outputCol=___
)

# COMMAND ----------

# TODO: StandardScalerを作成してください
# - inputCol: "raw_features"
# - outputCol: "features"
# - withStd: True
# - withMean: True
scaler = StandardScaler(
    inputCol=___,
    outputCol=___,
    withStd=___,
    withMean=___
)

# COMMAND ----------

# TODO: LogisticRegressionを作成してください
# - featuresCol: "features"
# - labelCol: "label"
# - maxIter: 100
# - regParam: 0.01
lr = LogisticRegression(
    featuresCol=___,
    labelCol=___,
    maxIter=___,
    regParam=___
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Pipelineの構築

# COMMAND ----------

# TODO: Pipelineを作成してください
# stages: [assembler, scaler, lr]
pipeline = Pipeline(stages=___)

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

# TODO: Pipelineを訓練データで学習してください
# ヒント: pipeline.fit(データフレーム)
model = pipeline.___

print("モデルの学習が完了しました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 予測の実行

# COMMAND ----------

# TODO: テストデータで予測を実行してください
# ヒント: model.transform(データフレーム)
predictions = model.___

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

# TODO: 各評価指標を計算してください
# ヒント: evaluator.evaluate(predictions)
auc = auc_evaluator.___
accuracy = acc_evaluator.___
f1 = f1_evaluator.___

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
    
    # TODO: パラメータを記録してください
    # ヒント: mlflow.log_param("パラメータ名", 値)
    mlflow.log_param("maxIter", ___)
    mlflow.log_param("regParam", ___)
    mlflow.log_param("algorithm", "LogisticRegression")
    
    # TODO: メトリクスを記録してください
    # ヒント: mlflow.log_metric("メトリクス名", 値)
    mlflow.log_metric("auc_roc", ___)
    mlflow.log_metric("accuracy", ___)
    mlflow.log_metric("f1_score", ___)
    
    # TODO: モデルを記録してください
    # ヒント: mlflow.spark.log_model(モデル, "アーティファクト名")
    mlflow.spark.log_model(___, "spark_model")
    
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

# TODO: AUCが最も高い結果を見つけてください
# ヒント: max()関数とlambdaを使用
best_result = max(results, key=lambda x: x["___"])

print("=" * 50)
print("ベストモデル")
print("=" * 50)
print(f"パラメータ: {best_result['params']}")
print(f"AUC: {best_result['auc']:.4f}")
print(f"Run ID: {best_result['run_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ベストモデルの登録(チャレンジ)

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

# ベストモデルを登録
best_run_id = best_result["run_id"]

# TODO: ベストモデルをレジストリに登録してください
# ヒント: mlflow.register_model("runs:/ランID/アーティファクトパス", モデル名)
try:
    result = mlflow.register_model(
        f"runs:/{best_run_id}/spark_model",
        ___
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
# MAGIC 
# MAGIC ### 次のステップ
# MAGIC 
# MAGIC 1. **異なるアルゴリズムを試す**: RandomForest, GBTClassifierなど
# MAGIC 2. **特徴量エンジニアリング**: 新しい特徴量の作成
# MAGIC 3. **クロスバリデーション**: より堅牢な評価
# MAGIC 4. **モデルのデプロイ**: バッチ推論やリアルタイムサービング

# COMMAND ----------

# MAGIC %md
# MAGIC ## おまけ: RandomForestで試してみよう(チャレンジ)

# COMMAND ----------

# RandomForestClassifierでモデルを構築
# コメントアウトを解除して実行してみてください

# from pyspark.ml.classification import RandomForestClassifier

# with mlflow.start_run(run_name="random_forest_challenge"):
#     assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
#     scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
#     rf = RandomForestClassifier(
#         featuresCol="features",
#         labelCol="label",
#         numTrees=100,
#         maxDepth=5
#     )
#     pipeline = Pipeline(stages=[assembler, scaler, rf])
    
#     model = pipeline.fit(train_df)
#     predictions = model.transform(test_df)
    
#     auc = auc_evaluator.evaluate(predictions)
#     accuracy = acc_evaluator.evaluate(predictions)
    
#     mlflow.log_param("algorithm", "RandomForest")
#     mlflow.log_param("numTrees", 100)
#     mlflow.log_param("maxDepth", 5)
#     mlflow.log_metric("test_auc", auc)
#     mlflow.log_metric("test_accuracy", accuracy)
#     mlflow.spark.log_model(model, "spark_model")
    
#     print(f"RandomForest AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
