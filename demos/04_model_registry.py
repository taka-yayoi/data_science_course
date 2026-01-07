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
# MAGIC ### 📍 カタログエクスプローラで確認
# MAGIC
# MAGIC 登録されたモデルはカタログエクスプローラから確認できます。
# MAGIC
# MAGIC 1. 左メニューの「カタログ」をクリック
# MAGIC 2. `ds_workshop_<ユーザー名>` > `ml` > `Models` を展開
# MAGIC 3. `wine_quality_logreg` をクリック
# MAGIC
# MAGIC **確認ポイント:**
# MAGIC - Version 1が表示されている
# MAGIC - 「Champion」エイリアスが設定されている
# MAGIC - メトリクス(ACC, F1, AUC)が記録されている

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Champion/Challengerパターン
# MAGIC
# MAGIC 本番環境でのモデル管理では**Champion/Challenger**パターンがよく使われます。
# MAGIC
# MAGIC | エイリアス | 役割 |
# MAGIC |------------|------|
# MAGIC | **Champion** | 本番運用中のモデル。推論APIやバッチ処理で使用される |
# MAGIC | **Challenger** | 評価中の新モデル。Championより優れていれば昇格 |
# MAGIC
# MAGIC **メリット:**
# MAGIC - 推論コードを変更せずにモデルを切り替え可能
# MAGIC - バージョン番号ではなく「役割」でモデルを参照できる
# MAGIC - ロールバックも容易(エイリアスを戻すだけ)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 改良版モデルの作成と登録

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
# MAGIC ### 📍 カタログエクスプローラで確認(2つのバージョン)
# MAGIC
# MAGIC 再度カタログエクスプローラで `wine_quality_logreg` を確認してください。
# MAGIC
# MAGIC **確認ポイント:**
# MAGIC - Version 1 と Version 2 が表示されている
# MAGIC - Version 1 に「Champion」エイリアス
# MAGIC - Version 2 に「Challenger」エイリアス
# MAGIC
# MAGIC これにより、本番用(Champion)と評価用(Challenger)のモデルが明確に区別されます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. モデル比較と本番昇格

# COMMAND ----------

# 2つのモデルの精度を比較
print("=" * 50)
print("モデル比較")
print("=" * 50)
print(f"Champion (v1):   ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
print(f"Challenger (v2): ACC={acc_v2:.4f}, F1={f1_v2:.4f}, AUC={auc_v2:.4f}")
print("=" * 50)

# Challengerが優れているか判定(同等以上なら新モデルを採用)
if auc_v2 >= auc:
    print("✅ Challenger がChampion以上の性能です！")
    promote = True
else:
    print("⚠️ Champion が引き続き最良です")
    promote = False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Challengerの本番昇格
# MAGIC
# MAGIC Challengerのほうが優れている場合、エイリアスを切り替えて本番昇格させます。
# MAGIC
# MAGIC **ポイント:** 推論コードは `@Champion` を参照しているため、コード変更なしでモデルが切り替わります。

# COMMAND ----------

# Challengerを新しいChampionに昇格
if promote:
    # 旧Championのエイリアスを削除(オプション)
    client.delete_registered_model_alias(name=MODEL_NAME, alias="Champion")
    
    # ChallengerをChampionに昇格
    client.set_registered_model_alias(name=MODEL_NAME, alias="Champion", version=model_version_v2.version)
    
    # Challengerエイリアスを削除
    client.delete_registered_model_alias(name=MODEL_NAME, alias="Challenger")
    
    print(f"🎉 Version {model_version_v2.version} を Champion に昇格しました！")
else:
    print("昇格はスキップされました")

# COMMAND ----------

# エイリアスの状態を確認
model_info_updated = client.get_registered_model(MODEL_NAME)
print("現在のエイリアス:")
if model_info_updated.aliases:
    for alias, version in model_info_updated.aliases.items():
        print(f"  - @{alias} -> Version {version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📍 カタログエクスプローラで確認(昇格後)
# MAGIC
# MAGIC カタログエクスプローラを更新して、エイリアスが変更されていることを確認してください。
# MAGIC
# MAGIC **確認ポイント:**
# MAGIC - Version 2 に「Champion」エイリアスが移動している
# MAGIC - Challengerエイリアスは削除されている

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Championモデルで推論
# MAGIC
# MAGIC エイリアスを使った推論の最大のメリットは、**コードを変更せずにモデルを切り替えられる**ことです。
# MAGIC
# MAGIC 以下のコードは昇格前後で全く同じですが、参照するモデルは自動的に新しいChampionに切り替わっています。

# COMMAND ----------

# Championモデルを読み込み(コードは変更なし！)
loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Champion")

# 全データで予測
pred_all = loaded_model.predict(X)
pred_proba_all = loaded_model.predict_proba(X)[:, 1]

print(f"✅ Championモデルで予測完了: {len(pred_all)}件")
print(f"   (参照先は自動的に最新のChampionに切り替わっています)")

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
# MAGIC ## 10. まとめ
# MAGIC
# MAGIC ### 学んだこと
# MAGIC
# MAGIC | 操作 | コード |
# MAGIC |------|--------|
# MAGIC | レジストリ設定 | `mlflow.set_registry_uri("databricks-uc")` |
# MAGIC | モデル登録 | `mlflow.register_model(model_uri, name)` |
# MAGIC | エイリアス設定 | `client.set_registered_model_alias(name, alias, version)` |
# MAGIC | エイリアス削除 | `client.delete_registered_model_alias(name, alias)` |
# MAGIC | モデル読み込み | `mlflow.sklearn.load_model("models:/name@alias")` |
# MAGIC
# MAGIC ### Champion/Challengerパターンのワークフロー
# MAGIC
# MAGIC 1. **初回デプロイ**: モデルを登録し、Championエイリアスを設定
# MAGIC 2. **改善検証**: 新モデルをChallengerとして登録
# MAGIC 3. **比較評価**: Champion vs Challengerの精度を比較
# MAGIC 4. **本番昇格**: Challengerが優れていれば、Championエイリアスを移動
# MAGIC 5. **推論継続**: `@Champion`を参照する推論コードは変更不要

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
