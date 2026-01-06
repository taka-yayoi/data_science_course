# Databricks notebook source
# MAGIC %md
# MAGIC # デモ2: SparkMLパイプラインの構築
# MAGIC 
# MAGIC このノートブックでは、SparkMLを使ってワインの品質を予測するモデルを構築します。
# MAGIC 
# MAGIC ## 学習目標
# MAGIC - SparkML Pipelineの基本概念を理解する
# MAGIC - Transformer / Estimator の使い方を学ぶ
# MAGIC - モデルの学習と評価を行う

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. データの準備

# COMMAND ----------

# Wine Qualityデータセットの読み込み
wine_df = spark.read.csv(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

# カラム名のスペースをアンダースコアに置換(SparkMLの要件)
for col_name in wine_df.columns:
    wine_df = wine_df.withColumnRenamed(col_name, col_name.replace(" ", "_"))

print("カラム名:")
print(wine_df.columns)

# COMMAND ----------

# データの確認
display(wine_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 特徴量とラベルの定義

# COMMAND ----------

# 品質スコアを二値分類に変換(6以上を「高品質」とする)
from pyspark.sql.functions import when, col

wine_df = wine_df.withColumn(
    "label",
    when(col("quality") >= 6, 1.0).otherwise(0.0)
)

# 特徴量カラムのリスト(qualityとlabelを除く)
feature_cols = [c for c in wine_df.columns if c not in ["quality", "label"]]
print(f"特徴量: {feature_cols}")

# COMMAND ----------

# ラベルの分布を確認
display(wine_df.groupBy("label").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 訓練データとテストデータの分割

# COMMAND ----------

# データを訓練用(80%)とテスト用(20%)に分割
train_df, test_df = wine_df.randomSplit([0.8, 0.2], seed=42)

print(f"訓練データ: {train_df.count()} 件")
print(f"テストデータ: {test_df.count()} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. SparkML Pipelineの構築

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 コンポーネントの定義

# COMMAND ----------

# Step 1: VectorAssembler - 複数の特徴量カラムを1つのベクトルに結合
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features"
)

# Step 2: StandardScaler - 特徴量の標準化
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Step 3: LogisticRegression - ロジスティック回帰モデル
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Pipelineの作成と学習

# COMMAND ----------

# Pipelineの構築
pipeline = Pipeline(stages=[assembler, scaler, lr])

print("Pipeline stages:")
for i, stage in enumerate(pipeline.getStages()):
    print(f"  {i+1}. {type(stage).__name__}")

# COMMAND ----------

# Pipelineの学習(fit)
# すべてのステージが順番に実行される
model = pipeline.fit(train_df)

print("モデルの学習が完了しました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. モデルの評価

# COMMAND ----------

# テストデータで予測
predictions = model.transform(test_df)

# 予測結果の確認
display(predictions.select("label", "prediction", "probability").limit(10))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# AUC-ROCの計算
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = auc_evaluator.evaluate(predictions)

# Accuracyの計算
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = acc_evaluator.evaluate(predictions)

# F1スコアの計算
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_evaluator.evaluate(predictions)

print("=" * 40)
print("モデル評価結果")
print("=" * 40)
print(f"AUC-ROC:  {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 混同行列の確認

# COMMAND ----------

# 混同行列の計算
confusion_matrix = predictions.groupBy("label", "prediction").count()
display(confusion_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. モデルの係数確認

# COMMAND ----------

# ロジスティック回帰モデルの係数を取得
lr_model = model.stages[-1]  # Pipelineの最後のステージ

print("モデルの係数:")
print(f"切片: {lr_model.intercept:.4f}")
print("\n特徴量の係数:")
for feature, coef in zip(feature_cols, lr_model.coefficients):
    print(f"  {feature}: {coef:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 新しいデータでの予測

# COMMAND ----------

# 新しいワインサンプルを作成(既存データから1行取得してデモ)
new_wine = test_df.limit(1).drop("label", "quality")
display(new_wine)

# COMMAND ----------

# 予測の実行
new_prediction = model.transform(new_wine)
display(new_prediction.select("prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 
# MAGIC このデモで学んだ内容:
# MAGIC 
# MAGIC 1. **VectorAssembler**: 複数の特徴量カラムを1つのベクトルに結合
# MAGIC 2. **StandardScaler**: 特徴量の標準化(平均0、標準偏差1)
# MAGIC 3. **LogisticRegression**: 二値分類モデル
# MAGIC 4. **Pipeline**: 複数のステージを連結してワークフローを管理
# MAGIC 5. **評価指標**: AUC-ROC、Accuracy、F1 Scoreによるモデル評価
# MAGIC 
# MAGIC ### Pipelineの利点
# MAGIC - 前処理からモデル学習まで一貫したワークフロー
# MAGIC - `fit()`で全ステージを順番に実行
# MAGIC - `transform()`で新しいデータに同じ処理を適用
# MAGIC - モデルの保存・読み込みが容易
# MAGIC 
# MAGIC 次のデモでは、MLflowを使った実験管理を行います。
