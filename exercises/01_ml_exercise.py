# Databricks notebook source
# MAGIC %md
# MAGIC # 実践演習: 機械学習ワークフロー
# MAGIC
# MAGIC この演習では、デモで学んだ内容を自分の手で実装します。
# MAGIC
# MAGIC **データセット:** Breast Cancer Wisconsin（乳がん診断データ）
# MAGIC - 30個の特徴量（細胞核の測定値）
# MAGIC - 2クラス分類（悪性/良性）
# MAGIC
# MAGIC **演習時間:** 約90分
# MAGIC
# MAGIC **ヒント:** 困ったらデモノートブックを参照してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習0: MLflowアップグレード（必要な場合）
# MAGIC
# MAGIC サーバレスv2を使用している場合は、以下のセルのコメントを外して実行してください。

# COMMAND ----------

# サーバレスv2の場合は以下のコメントを外して実行
# %pip install --upgrade mlflow -q

# COMMAND ----------

# 上記を実行した場合は、このセルも実行
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習1: 環境セットアップ
# MAGIC
# MAGIC Unity Catalog上にカタログとスキーマを作成してください。

# COMMAND ----------

# ライブラリのインポート（実行するだけでOK）
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from mlflow.models import infer_signature
import time

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# COMMAND ----------

# ユーザー名を取得（実行するだけでOK）
username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')
print(f"ユーザー名: {clean_username}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題1-1: カタログとスキーマの定義
# MAGIC
# MAGIC 以下の変数を設定してください。
# MAGIC - カタログ名: `exercise_{ユーザー名}`
# MAGIC - スキーマ名: `ml`

# COMMAND ----------

# TODO: カタログ名とスキーマ名を定義
CATALOG = ___  # ヒント: f"exercise_{clean_username}"
SCHEMA = ___   # ヒント: "ml"

# モデル名とテーブル名（上記を設定すれば自動で決まります）
MODEL_NAME = f"{CATALOG}.{SCHEMA}.breast_cancer_classifier"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.breast_cancer_predictions"

print(f"カタログ: {CATALOG}")
print(f"スキーマ: {SCHEMA}")
print(f"モデル名: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題1-2: カタログとスキーマの作成
# MAGIC
# MAGIC SQLでカタログとスキーマを作成してください。

# COMMAND ----------

# TODO: カタログを作成（存在しない場合のみ）
spark.sql(f"___")

# TODO: スキーマを作成（存在しない場合のみ）
spark.sql(f"___")

print(f"✅ カタログとスキーマを作成しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習2: データの準備
# MAGIC
# MAGIC Breast Cancerデータセットを読み込み、探索します。

# COMMAND ----------

# データの読み込み（実行するだけでOK）
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target  # 0=悪性, 1=良性

print(f"データサイズ: {df.shape}")
print(f"ターゲット分布:\n{df['target'].value_counts()}")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題2-1: データの分割
# MAGIC
# MAGIC データを特徴量(X)とターゲット(y)に分割し、訓練データとテストデータに分けてください。
# MAGIC
# MAGIC - テストサイズ: 20%
# MAGIC - random_state: 42

# COMMAND ----------

# 特徴量とターゲットを分離
X = df.drop('target', axis=1)
y = df['target']

# TODO: train_test_splitでデータを分割
X_train, X_test, y_train, y_test = train_test_split(
    ___,  # 特徴量
    ___,  # ターゲット
    test_size=___,     # テストサイズ
    random_state=___   # 乱数シード
)

print(f"訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習3: パイプライン構築とMLflowトラッキング
# MAGIC
# MAGIC scikit-learnのパイプラインを構築し、MLflowで実験を記録します。

# COMMAND ----------

# MLflow設定（実行するだけでOK）
mlflow.set_registry_uri("databricks-uc")
EXPERIMENT_NAME = f"/Users/{username}/breast_cancer_exercise"
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
print(f"実験名: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題3-1: ロジスティック回帰パイプラインの構築
# MAGIC
# MAGIC StandardScalerとLogisticRegressionを組み合わせたパイプラインを作成してください。
# MAGIC
# MAGIC - LogisticRegressionのパラメータ: C=1.0, max_iter=1000, random_state=42

# COMMAND ----------

# TODO: パイプラインを構築
pipeline_lr = Pipeline([
    (___),  # ヒント: ("scaler", StandardScaler())
    (___)   # ヒント: ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

print("パイプライン構築完了")
print(pipeline_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題3-2: MLflowでの実験記録
# MAGIC
# MAGIC モデルを学習し、MLflowで以下を記録してください:
# MAGIC - パラメータ: model_type, C
# MAGIC - メトリクス: accuracy, f1_score, roc_auc
# MAGIC - モデル: sklearn形式で保存

# COMMAND ----------

# TODO: MLflowで実験を記録
with mlflow.start_run(run_name="LogisticRegression_C1.0") as run:
    # モデル学習
    pipeline_lr.fit(X_train, y_train)
    
    # 予測
    y_pred = pipeline_lr.predict(X_test)
    y_proba = pipeline_lr.predict_proba(X_test)[:, 1]
    
    # メトリクス計算
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # TODO: パラメータをログ
    mlflow.log_params({
        ___  # ヒント: "model_type": "LogisticRegression", "C": 1.0
    })
    
    # TODO: メトリクスをログ
    mlflow.log_metrics({
        ___  # ヒント: "accuracy": acc, "f1_score": f1, "roc_auc": auc
    })
    
    # シグネチャを推論
    signature = infer_signature(X_train, pipeline_lr.predict(X_train))
    
    # TODO: モデルをログ
    mlflow.sklearn.log_model(
        sk_model=___,        # パイプライン
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(2)
    )
    
    run_id_lr = run.info.run_id
    print(f"Run ID: {run_id_lr}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習4: ハイパーパラメータ比較
# MAGIC
# MAGIC 異なるパラメータでモデルを学習し、結果を比較します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題4-1: 複数のCの値で実験
# MAGIC
# MAGIC LogisticRegressionのCパラメータを変えて、複数の実験を実行してください。
# MAGIC
# MAGIC - C = [0.01, 0.1, 1.0, 10.0] の4パターン

# COMMAND ----------

# TODO: 異なるCの値で実験を実行
C_values = [0.01, 0.1, 1.0, 10.0]
results = []

for C in C_values:
    with mlflow.start_run(run_name=f"LogisticRegression_C{C}") as run:
        # TODO: パイプラインを構築（Cの値を変える）
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=___, max_iter=1000, random_state=42))
        ])
        
        # 学習
        pipeline.fit(X_train, y_train)
        
        # 予測とメトリクス
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # ログ
        mlflow.log_params({"model_type": "LogisticRegression", "C": C})
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": auc})
        
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, "model", signature=signature)
        
        results.append({
            "C": C,
            "run_id": run.info.run_id,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc
        })
        
        print(f"C={C}: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# COMMAND ----------

# 結果の比較（実行するだけでOK）
results_df = pd.DataFrame(results)
display(results_df.sort_values("roc_auc", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📍 MLflow UIで確認
# MAGIC
# MAGIC 左メニューの「エクスペリメント」から `breast_cancer_exercise` を開き、
# MAGIC 4つの実験結果を比較してください。
# MAGIC
# MAGIC - どのCの値が最も良い結果でしたか？
# MAGIC - チャートビューでメトリクスを可視化してみましょう

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習5: 最良モデルをUnity Catalogに登録
# MAGIC
# MAGIC 最も良い結果のモデルをUnity Catalogに登録します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題5-1: 最良モデルの特定と登録

# COMMAND ----------

# 最良のモデルを特定（実行するだけでOK）
best_result = max(results, key=lambda x: x["roc_auc"])
best_run_id = best_result["run_id"]
best_C = best_result["C"]
print(f"最良モデル: C={best_C}, AUC={best_result['roc_auc']:.4f}")
print(f"Run ID: {best_run_id}")

# COMMAND ----------

# TODO: 最良モデルをUnity Catalogに登録
model_uri = f"runs:/{best_run_id}/model"

# ヒント: mlflow.register_model(model_uri=..., name=...)
model_version = mlflow.register_model(
    model_uri=___,
    name=___
)

time.sleep(3)
print(f"✅ モデル登録完了: {MODEL_NAME} v{model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題5-2: Championエイリアスの設定

# COMMAND ----------

# TODO: Championエイリアスを設定
# ヒント: client.set_registered_model_alias(name=..., alias=..., version=...)
client.set_registered_model_alias(
    name=___,
    alias=___,
    version=___
)

print(f"✅ Championエイリアスを設定しました")

# COMMAND ----------

# 登録モデルの確認（実行するだけでOK）
model_info = client.get_registered_model(MODEL_NAME)
print(f"モデル名: {model_info.name}")
if model_info.aliases:
    print("エイリアス:")
    for alias, version in model_info.aliases.items():
        print(f"  - @{alias} -> Version {version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📍 カタログエクスプローラで確認
# MAGIC
# MAGIC カタログエクスプローラで `exercise_{ユーザー名}` > `ml` > `Models` を開き、
# MAGIC 登録したモデルとChampionエイリアスを確認してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習6: Championモデルで推論

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題6-1: モデルの読み込みと推論

# COMMAND ----------

# TODO: Championモデルを読み込み
# ヒント: mlflow.sklearn.load_model("models:/モデル名@Champion")
loaded_model = mlflow.sklearn.load_model(f"___")

# 全データで予測
predictions = loaded_model.predict(X)
probabilities = loaded_model.predict_proba(X)[:, 1]

print(f"✅ 予測完了: {len(predictions)}件")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題6-2: 推論結果をDeltaテーブルに保存

# COMMAND ----------

# 予測結果をDataFrameに変換（実行するだけでOK）
pred_df = pd.DataFrame({
    "sample_id": np.arange(len(df)),
    "prediction": predictions.astype(int),
    "probability": probabilities.astype(float),
    "actual": y.values
})

# Sparkに変換
pred_sdf = spark.createDataFrame(pred_df)
display(pred_sdf)

# COMMAND ----------

# TODO: テーブルとして保存
# ヒント: pred_sdf.write.mode("overwrite").saveAsTable(テーブル名)
pred_sdf.write.mode(___).saveAsTable(___)

print(f"✅ テーブル保存完了: {PRED_TABLE}")

# COMMAND ----------

# 保存したテーブルを確認（実行するだけでOK）
display(spark.table(PRED_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習完了！🎉
# MAGIC
# MAGIC おめでとうございます！以下の内容を実践しました:
# MAGIC
# MAGIC 1. ✅ Unity Catalogにカタログ・スキーマを作成
# MAGIC 2. ✅ データの読み込みと分割
# MAGIC 3. ✅ scikit-learnパイプラインの構築
# MAGIC 4. ✅ MLflowで実験をトラッキング
# MAGIC 5. ✅ ハイパーパラメータ比較
# MAGIC 6. ✅ 最良モデルをUnity Catalogに登録
# MAGIC 7. ✅ Championエイリアスを設定
# MAGIC 8. ✅ モデルを読み込んで推論
# MAGIC 9. ✅ 結果をDeltaテーブルに保存

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ（必要に応じて実行）

# COMMAND ----------

# # モデルとテーブルの削除
# client.delete_registered_model(MODEL_NAME)
# spark.sql(f"DROP TABLE IF EXISTS {PRED_TABLE}")
# spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
# spark.sql(f"DROP CATALOG IF EXISTS {CATALOG} CASCADE")
# print("クリーンアップ完了")
