# Databricks notebook source
# MAGIC %md
# MAGIC # デモ1: Databricks ML環境の確認
# MAGIC 
# MAGIC このノートブックでは、Databricks Free EditionのML環境を確認します。
# MAGIC 
# MAGIC ## 学習目標
# MAGIC - Databricksワークスペースの構成を理解する
# MAGIC - 利用可能なMLライブラリを確認する
# MAGIC - サーバーレスコンピュートでの機械学習環境を理解する

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 環境情報の確認

# COMMAND ----------

# Sparkのバージョン確認
print(f"Spark バージョン: {spark.version}")

# COMMAND ----------

# Pythonのバージョン確認
import sys
print(f"Python バージョン: {sys.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 主要MLライブラリの確認

# COMMAND ----------

# インストール済みのMLライブラリを確認
import importlib

libraries = [
    ("sklearn", "scikit-learn"),
    ("mlflow", "MLflow"),
    ("pandas", "pandas"),
    ("numpy", "NumPy"),
    ("xgboost", "XGBoost"),
]

print("=" * 50)
print("利用可能なMLライブラリ")
print("=" * 50)

for module_name, display_name in libraries:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "バージョン不明")
        print(f"✓ {display_name}: {version}")
    except ImportError:
        print(f"✗ {display_name}: インストールされていません")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Free Editionサーバーレス環境の特徴
# MAGIC 
# MAGIC | 項目 | 説明 |
# MAGIC |------|------|
# MAGIC | 計算リソース | サーバーレスコンピュート(自動管理) |
# MAGIC | 利用可能なML | scikit-learn, XGBoost, pandas等 |
# MAGIC | 制限事項 | SparkML(pyspark.ml)は利用不可 |
# MAGIC | MLflow | 完全サポート |
# MAGIC | Unity Catalog | 利用可能 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MLflowの確認

# COMMAND ----------

import mlflow

# サーバーレス環境用: レジストリURIを明示的に設定
mlflow.set_registry_uri("databricks-uc")

# MLflowのトラッキングURIを確認
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

# 現在のユーザーを取得
username = spark.sql("SELECT current_user()").collect()[0][0]
print(f"現在のユーザー: {username}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. カタログとスキーマの確認

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 利用可能なカタログを確認
# MAGIC SHOW CATALOGS

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 現在のカタログとスキーマを確認
# MAGIC SELECT current_catalog(), current_schema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. サンプルデータの確認

# COMMAND ----------

# Databricksのサンプルデータセットを確認
# Wine Qualityデータセットを使用
wine_df = spark.read.csv(
    "/databricks-datasets/wine-quality/winequality-red.csv",
    header=True,
    inferSchema=True,
    sep=";"
)

print(f"レコード数: {wine_df.count()}")
print(f"カラム数: {len(wine_df.columns)}")

# COMMAND ----------

# スキーマの確認
wine_df.printSchema()

# COMMAND ----------

# データのプレビュー
display(wine_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Spark DataFrameからpandasへの変換

# COMMAND ----------

# pandasに変換(小規模データの場合)
pdf = wine_df.toPandas()
print(f"pandas DataFrame: {pdf.shape}")
pdf.head()

# COMMAND ----------

# 基本統計量
pdf.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 
# MAGIC このデモで確認した内容:
# MAGIC 
# MAGIC 1. **Sparkバージョン**: Databricksが提供する最適化されたSpark環境
# MAGIC 2. **MLライブラリ**: scikit-learn、MLflow、XGBoost等が事前インストール済み
# MAGIC 3. **MLflow統合**: DatabricksワークスペースにMLflowが統合されている
# MAGIC 4. **Unity Catalog**: データガバナンスのためのカタログ機能
# MAGIC 5. **サンプルデータ**: `/databricks-datasets/`に各種データセットが用意されている
# MAGIC 
# MAGIC ### Free Editionでの機械学習ワークフロー
# MAGIC 
# MAGIC ```
# MAGIC Spark DataFrame → .toPandas() → scikit-learn → MLflow → Unity Catalog
# MAGIC ```
# MAGIC 
# MAGIC - データの読み込み・書き込みはSparkを使用
# MAGIC - 機械学習はscikit-learnで実行
# MAGIC - 実験管理・モデル管理はMLflowで実行
# MAGIC 
# MAGIC 次のデモでは、scikit-learnを使ったモデル開発を行います。
