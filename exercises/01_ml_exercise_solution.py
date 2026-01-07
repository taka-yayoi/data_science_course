# Databricks notebook source
# MAGIC %md
# MAGIC # 実践演習: 機械学習ワークフロー【模範解答】
# MAGIC
# MAGIC この演習では、デモで学んだ内容を自分の手で実装します。
# MAGIC
# MAGIC **データセット:** Breast Cancer Wisconsin（乳がん診断データ）
# MAGIC - 30個の特徴量（細胞核の測定値）
# MAGIC - 2クラス分類: 0 = malignant（悪性）、1 = benign（良性）
# MAGIC
# MAGIC **コンピュート:** サーバレスv4を使用してください（右の環境パネルから設定）
# MAGIC
# MAGIC **演習時間:** 約90分

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💡 演習のヒント
# MAGIC
# MAGIC ### コード補完を活用しよう
# MAGIC 穴埋め箇所（`___`）を消して少し待つと、**オートコンプリーション**が候補を表示します。
# MAGIC Tabキーで補完を確定できます。
# MAGIC
# MAGIC ### Databricks Assistantを活用しよう
# MAGIC 右サイドバーの「Assistant」アイコンをクリックすると、AIアシスタントに質問できます。
# MAGIC
# MAGIC ### デモノートブックを参照
# MAGIC `demos/` フォルダのノートブックにコード例があります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習0: MLflowアップグレード（必要な場合）

# COMMAND ----------

# サーバレスv2の場合は以下のコメントを外して実行
# %pip install --upgrade mlflow -q

# COMMAND ----------

# 上記を実行した場合は、このセルも実行
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習1: 環境セットアップ

# COMMAND ----------

# ライブラリのインポート
import numpy as np
import pandas as pd
import plotly.express as px
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

# カタログとスキーマの設定
username = spark.sql("SELECT current_user()").collect()[0][0]
clean_username = username.split('@')[0].replace('.', '_').replace('-', '_')

CATALOG = f"exercise_{clean_username}"
SCHEMA = "ml"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.breast_cancer_classifier"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.breast_cancer_predictions"

# カタログとスキーマを作成
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"✅ 環境セットアップ完了")
print(f"   カタログ: {CATALOG}")
print(f"   スキーマ: {SCHEMA}")
print(f"   モデル名: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習2: データの準備とEDA

# COMMAND ----------

# データの読み込み
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})

print(f"データサイズ: {df.shape}")
print(f"\nターゲット分布:")
print(df['target_name'].value_counts())
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題2-1: 基本統計量の確認

# COMMAND ----------

# 【解答】describe()で基本統計量を表示
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題2-2: ターゲット分布の可視化

# COMMAND ----------

# 【解答】ターゲット分布を棒グラフで表示
fig = px.histogram(
    df,
    x="target_name",
    color="target_name",
    title="ターゲット分布（悪性 vs 良性）"
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題2-3: 特徴量の相関確認

# COMMAND ----------

# 主要な特徴量の散布図
fig = px.scatter(
    df,
    x="mean radius",
    y="mean texture",
    color="target_name",
    title="主要特徴量の散布図",
    labels={"mean radius": "平均半径", "mean texture": "平均テクスチャ"}
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題2-4: データの分割

# COMMAND ----------

# 特徴量とターゲットを分離
X = df.drop(['target', 'target_name'], axis=1)
y = df['target']

# 【解答】train_test_splitでデータを分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"訓練データ: {len(X_train)}件")
print(f"テストデータ: {len(X_test)}件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習3: パイプライン構築とMLflowトラッキング

# COMMAND ----------

# MLflow設定
mlflow.set_registry_uri("databricks-uc")
EXPERIMENT_NAME = f"/Users/{username}/breast_cancer_exercise"
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# エクスペリメントへのリンクを表示
displayHTML(f"""
<p>✅ エクスペリメントを設定しました</p>
<p>👉 <a href="#mlflow/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}" target="_blank">
MLflow エクスペリメントを開く: {EXPERIMENT_NAME}</a></p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題3-1: ロジスティック回帰パイプラインの構築

# COMMAND ----------

# 【解答】パイプラインを構築
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

print("パイプライン構築完了")
print(pipeline_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題3-2: MLflowでの実験記録

# COMMAND ----------

# 【解答】MLflowで実験を記録
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
    
    # パラメータをログ
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "C": 1.0
    })
    
    # メトリクスをログ
    mlflow.log_metrics({
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc
    })
    
    # シグネチャを推論
    signature = infer_signature(X_train, pipeline_lr.predict(X_train))
    
    # モデルをログ
    mlflow.sklearn.log_model(
        sk_model=pipeline_lr,
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題4-1: 複数のCの値で実験

# COMMAND ----------

# 【解答】異なるCの値で実験を実行
C_values = [0.01, 0.1, 1.0, 10.0]
results = []

for C in C_values:
    with mlflow.start_run(run_name=f"LogisticRegression_C{C}") as run:
        # パイプラインを構築
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=1000, random_state=42))
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

# 結果の比較
results_df = pd.DataFrame(results)
display(results_df.sort_values("roc_auc", ascending=False))

# COMMAND ----------

# MLflow UIへのリンク
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
displayHTML(f"""
<h3>📊 MLflow UIで結果を確認</h3>
<p>👉 <a href="#mlflow/experiments/{experiment.experiment_id}" target="_blank">
エクスペリメントを開く: {EXPERIMENT_NAME}</a></p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習5: 最良モデルをUnity Catalogに登録

# COMMAND ----------

# 最良のモデルを特定
best_result = max(results, key=lambda x: x["roc_auc"])
best_run_id = best_result["run_id"]
best_C = best_result["C"]
print(f"最良モデル: C={best_C}, AUC={best_result['roc_auc']:.4f}")
print(f"Run ID: {best_run_id}")

# COMMAND ----------

# 【解答】最良モデルをUnity Catalogに登録
model_uri = f"runs:/{best_run_id}/model"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

time.sleep(3)
print(f"✅ モデル登録完了: {MODEL_NAME} v{model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 課題5-2: Championエイリアスの設定

# COMMAND ----------

# 【解答】Championエイリアスを設定
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=model_version.version
)

print(f"✅ Championエイリアスを設定しました")

# COMMAND ----------

# 登録モデルの確認
model_info = client.get_registered_model(MODEL_NAME)
print(f"モデル名: {model_info.name}")
if model_info.aliases:
    print("エイリアス:")
    for alias, version in model_info.aliases.items():
        print(f"  - @{alias} -> Version {version}")

# COMMAND ----------

# カタログエクスプローラへのリンク
displayHTML(f"""
<h3>📦 カタログエクスプローラで確認</h3>
<p>👉 <a href="/explore/data/models/{CATALOG}/{SCHEMA}/{MODEL_NAME.split('.')[-1]}" target="_blank">
モデルを開く: {MODEL_NAME}</a></p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習6: Championモデルで推論

# COMMAND ----------

# 【解答】Championモデルを読み込み
loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Champion")

# 全データで予測
predictions = loaded_model.predict(X)
probabilities = loaded_model.predict_proba(X)[:, 1]

print(f"✅ 予測完了: {len(predictions)}件")

# COMMAND ----------

# 予測結果をDataFrameに変換
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

# 【解答】テーブルとして保存
pred_sdf.write.mode("overwrite").saveAsTable(PRED_TABLE)

print(f"✅ テーブル保存完了: {PRED_TABLE}")

# COMMAND ----------

# 保存したテーブルへのリンク
displayHTML(f"""
<h3>📊 推論結果テーブル</h3>
<p>👉 <a href="/explore/data/{CATALOG}/{SCHEMA}/{PRED_TABLE.split('.')[-1]}" target="_blank">
テーブルを開く: {PRED_TABLE}</a></p>
""")

# COMMAND ----------

# 保存したテーブルを確認
display(spark.table(PRED_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 演習完了！🎉

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 追加課題（チャレンジ）【解答例】

# COMMAND ----------

# MAGIC %md
# MAGIC ### チャレンジ1: RandomForestで実験

# COMMAND ----------

# 【解答例】RandomForestで実験
with mlflow.start_run(run_name="RandomForest_n100") as run:
    pipeline_rf = Pipeline([
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline_rf.fit(X_train, y_train)
    
    y_pred_rf = pipeline_rf.predict(X_test)
    y_proba_rf = pipeline_rf.predict_proba(X_test)[:, 1]
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    
    mlflow.log_params({"model_type": "RandomForest", "n_estimators": 100})
    mlflow.log_metrics({"accuracy": acc_rf, "f1_score": f1_rf, "roc_auc": auc_rf})
    
    signature = infer_signature(X_train, pipeline_rf.predict(X_train))
    mlflow.sklearn.log_model(pipeline_rf, "model", signature=signature)
    
    run_id_rf = run.info.run_id
    print(f"RandomForest: ACC={acc_rf:.4f}, F1={f1_rf:.4f}, AUC={auc_rf:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### チャレンジ2: 混同行列の可視化

# COMMAND ----------

# 【解答例】混同行列の可視化
import plotly.figure_factory as ff

cm = confusion_matrix(y_test, y_pred)
fig = ff.create_annotated_heatmap(
    cm,
    x=['Predicted Malignant', 'Predicted Benign'],
    y=['Actual Malignant', 'Actual Benign'],
    colorscale='Blues'
)
fig.update_layout(title="混同行列")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### チャレンジ3: Challengerモデルの登録と昇格

# COMMAND ----------

# 【解答例】RandomForestをChallengerとして登録
model_uri_rf = f"runs:/{run_id_rf}/model"
model_version_rf = mlflow.register_model(model_uri=model_uri_rf, name=MODEL_NAME)

time.sleep(3)

# Challengerエイリアスを設定
client.set_registered_model_alias(name=MODEL_NAME, alias="Challenger", version=model_version_rf.version)
print(f"✅ Challenger登録: v{model_version_rf.version}")

# COMMAND ----------

# 比較と昇格判定
print("=" * 50)
print("モデル比較")
print("=" * 50)
print(f"Champion (LogReg): AUC={best_result['roc_auc']:.4f}")
print(f"Challenger (RF):   AUC={auc_rf:.4f}")
print("=" * 50)

if auc_rf >= best_result['roc_auc']:
    # 旧Championのエイリアスを削除
    client.delete_registered_model_alias(name=MODEL_NAME, alias="Champion")
    
    # ChallengerをChampionに昇格
    client.set_registered_model_alias(name=MODEL_NAME, alias="Champion", version=model_version_rf.version)
    
    # Challengerエイリアスを削除
    client.delete_registered_model_alias(name=MODEL_NAME, alias="Challenger")
    
    print(f"🎉 RandomForest (v{model_version_rf.version}) を Champion に昇格しました！")
else:
    print("昇格はスキップされました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ
# MAGIC
# MAGIC 演習で作成したモデルとテーブルのみ削除します。

# COMMAND ----------

# # 演習で作成したモデルとテーブルの削除
# client.delete_registered_model(MODEL_NAME)
# spark.sql(f"DROP TABLE IF EXISTS {PRED_TABLE}")
# print("クリーンアップ完了")
