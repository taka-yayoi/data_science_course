# Databricks notebook source
# MAGIC %md
# MAGIC # 共通設定
# MAGIC このノートブックは他のノートブックから呼び出される共通設定です。

# COMMAND ----------

# カタログ・スキーマの設定
# Free Editionではデフォルトカタログを使用
CATALOG_NAME = "main"
SCHEMA_NAME = "default"

# 実験名のプレフィックス
EXPERIMENT_PREFIX = "/Users"

# COMMAND ----------

# ユーザー名を取得してユニークなスキーマ名を生成
import re

# 現在のユーザー名を取得
username = spark.sql("SELECT current_user()").collect()[0][0]
# メールアドレスからユーザー名部分を抽出してクリーンアップ
clean_username = re.sub(r'[^a-zA-Z0-9]', '_', username.split('@')[0])

# ユーザー固有のスキーマ名
USER_SCHEMA = f"ds_workshop_{clean_username}"

print(f"ユーザー: {username}")
print(f"スキーマ: {USER_SCHEMA}")

# COMMAND ----------

# スキーマの作成(存在しない場合)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{USER_SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {USER_SCHEMA}")

print(f"カタログ '{CATALOG_NAME}' のスキーマ '{USER_SCHEMA}' を使用します")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定完了
# MAGIC 以下の変数が利用可能です:
# MAGIC - `CATALOG_NAME`: カタログ名
# MAGIC - `USER_SCHEMA`: ユーザー固有のスキーマ名
# MAGIC - `username`: 現在のユーザー名
