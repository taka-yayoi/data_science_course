# Databricks notebook source
# MAGIC %md
# MAGIC # サーバレスで MMF を 10 分で回す: 5系列 × 5モデル 縮小版
# MAGIC
# MAGIC `databricks-industry-solutions/many-model-forecasting` の
# MAGIC `examples/daily/local_univariate_daily.ipynb` をベースに、データ規模とモデル数を絞って
# MAGIC 実行時間を圧縮した実習用ノートブック。`mmf_sa.run_forecast` はそのまま使う。
# MAGIC
# MAGIC 元サンプルからの差分:
# MAGIC
# MAGIC - 99系列 → 5系列に縮小、各系列の履歴も先頭 400 点に切り詰め
# MAGIC - active_models を 16 → 5 個に絞る (性格の違うモデルだけ残す)
# MAGIC - backtest_length 30 → 20、stride 10 → 10 で window 数を 3 → 2 に
# MAGIC - 各ステップに可視化と解釈ガイドを追加
# MAGIC
# MAGIC 想定実行時間: サーバレス standard で 10 分前後 (初回ライブラリインストール込み)
# MAGIC
# MAGIC リポジトリの clone は不要。MMFパッケージを pip 経由で直接入れる方式にしてある。
# MAGIC
# MAGIC ## このノートブックで体感してほしいこと
# MAGIC
# MAGIC MMFの核心は「多数の時系列に対して、系列ごとに最適なモデルが違う」という事実を、
# MAGIC 自動的に発見してくれる点にある。本ノートブックでは以下の流れで体感する。
# MAGIC
# MAGIC 1. 5本の時系列を可視化 (それぞれ違う性格を持っている)
# MAGIC 2. 5本に対して5モデルを一括適用 (これがMMFの仕事)
# MAGIC 3. 全体ランキングを見る (どのモデルが「平均的に」強いか)
# MAGIC 4. 系列ごとのチャンピオンを見る (実は系列によって勝者が違う!)
# MAGIC 5. 1系列を選んで、モデルごとの予測の癖を可視化

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ1: ライブラリのインストール
# MAGIC
# MAGIC MMFパッケージ (`mmf_sa`) は PyPI 公開されていないので、GitHub から直接インストールする。
# MAGIC `[local]` extras を指定して `statsforecast` などの local モデル依存をまとめて入れる。
# MAGIC データセット取得用の `datasetsforecast` も合わせて入れる。

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git" \
# MAGIC   datasetsforecast==0.0.8
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ2: パラメーター設定
# MAGIC
# MAGIC `CATALOG` / `SCHEMA` は自分の環境に書き換える。

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "m4_mini"
TRAIN_TABLE = "daily_train_5"
EVAL_TABLE = "daily_evaluation_5"
SCORE_TABLE = "daily_scoring_5"
USE_CASE = "m4_daily_mini"
EXPERIMENT_PATH = "/Shared/mmf_mini_experiment"

N_SERIES = 5             # 並列実行する系列数
HISTORY_LIMIT = 400      # 各系列の学習履歴を切り詰める長さ

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ3: M4 Daily から 5 系列を抽出
# MAGIC
# MAGIC 元サンプルは 99 系列を扱うが、本ノートブックでは欠損のない先頭 5 系列だけに絞る。
# MAGIC 履歴も先頭 400 点に切り詰めて、AutoARIMA など fit が重いモデルの計算量を抑える。
# MAGIC
# MAGIC `datasetsforecast.m4.M4.load()` が返す `ds` は系列内 1 始まりの整数なので、
# MAGIC 起点日を決めて日数オフセットで明示的に datetime 化する。

# COMMAND ----------

from datasetsforecast.m4 import M4
import pandas as pd

train_df, _, _ = M4.load(directory='/tmp/m4', group='Daily')

# 起点日 + 日数オフセットで datetime 化 (pd.to_datetime を直接当てると nanoseconds 解釈になる)
DATE_ORIGIN = pd.Timestamp("2020-01-01")
train_df['ds'] = DATE_ORIGIN + pd.to_timedelta(train_df['ds'].astype(int) - 1, unit='D')

counts = train_df.groupby('unique_id').size()
target_ids = counts[counts >= HISTORY_LIMIT].index[:N_SERIES].tolist()
print(f"selected unique_ids: {target_ids}")

# head() を使う。tail だと履歴長の違う系列で最終日が揃わず、品質チェックで落ちる場合がある。
small = (
    train_df[train_df['unique_id'].isin(target_ids)]
    .sort_values(['unique_id', 'ds'])
    .groupby('unique_id', group_keys=False)
    .head(HISTORY_LIMIT)
    .reset_index(drop=True)
)
print(f"rows: {len(small)} (= {N_SERIES} 系列 × {HISTORY_LIMIT} 点)")
print(f"date range: {small['ds'].min()} -- {small['ds'].max()}")

# COMMAND ----------

(
    spark.createDataFrame(small)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.{TRAIN_TABLE}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 抽出した5系列を可視化
# MAGIC
# MAGIC 各系列の長さ・スケール・季節性パターンが違うことを目で確認しておく。
# MAGIC この「系列ごとの個性」が、最適モデルが系列で違う理由の出発点になる。

# COMMAND ----------

import plotly.express as px

fig = px.line(
    small,
    x="ds",
    y="y",
    facet_row="unique_id",
    height=900,
    title=f"M4 Daily から抽出した {N_SERIES} 系列 (各 {HISTORY_LIMIT} 点)",
)
fig.update_yaxes(matches=None, showticklabels=True)  # 系列ごとに y 軸スケール独立
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ4: active_models の絞り込み
# MAGIC
# MAGIC 元サンプルは 16 モデル全部当てるが、本ノートブックは性格の違う 5 モデルだけに絞る。
# MAGIC これでも系列ごとにチャンピオンが分かれることが体感できる。
# MAGIC
# MAGIC - Baseline 系 2 つ — 単純な比較対象
# MAGIC   - `BaselineNaive`: 直前の値をそのまま返す
# MAGIC   - `BaselineSeasonalNaive`: 1 週間前の値をそのまま返す
# MAGIC - Auto 系 3 つ — それぞれモデル形式が違う
# MAGIC   - `AutoArima`: 自己回帰移動平均
# MAGIC   - `AutoETS`: 指数平滑
# MAGIC   - `AutoTheta`: Theta 法
# MAGIC
# MAGIC 重い AutoTbats、間欠需要向けの Croston 系、ワーカー側初期化に失敗する SKTimeProphet は外す。

# COMMAND ----------

active_models = [
    "StatsForecastBaselineNaive",
    "StatsForecastBaselineSeasonalNaive",
    "StatsForecastAutoArima",
    "StatsForecastAutoETS",
    "StatsForecastAutoTheta",
]

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ5: run_forecast の実行
# MAGIC
# MAGIC `mmf_sa.run_forecast` を呼ぶ。内部で applyInPandas ベースの並列処理が走り、
# MAGIC 5 系列 × 5 モデル × 2 backtest window のバックテストと、5 モデルの未来予測がまとめて実行される。
# MAGIC 結果は `evaluation_output` と `scoring_output` の 2 つの Delta テーブルに書き出される。
# MAGIC
# MAGIC `backtest_length=20` / `stride=10` で window 数を 2 に絞っている (元サンプルは 6)。
# MAGIC `data_quality_check=False` にして MMF 内蔵の品質チェックをバイパスする
# MAGIC (絞り込み済みデータで自分で担保する前提)。

# COMMAND ----------

import time
from mmf_sa import run_forecast

t0 = time.time()

run_forecast(
    spark=spark,
    train_data=f"{CATALOG}.{SCHEMA}.{TRAIN_TABLE}",
    scoring_data=f"{CATALOG}.{SCHEMA}.{TRAIN_TABLE}",
    scoring_output=f"{CATALOG}.{SCHEMA}.{SCORE_TABLE}",
    evaluation_output=f"{CATALOG}.{SCHEMA}.{EVAL_TABLE}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=10,
    backtest_length=20,
    stride=10,
    metric="smape",
    train_predict_ratio=2,
    data_quality_check=False,
    resample=False,
    active_models=active_models,
    experiment_path=EXPERIMENT_PATH,
    use_case_name=USE_CASE,
)

print(f"elapsed: {time.time() - t0:.1f} sec")

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ6: 結果テーブルの確認
# MAGIC
# MAGIC `actual` / `forecast` は配列型 (長さ 10) で入っている。1 行が「1 系列 × 1 モデル × 1 backtest window」に対応する。

# COMMAND ----------

display(spark.read.table(f"{CATALOG}.{SCHEMA}.{EVAL_TABLE}").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ7: 全体ランキング (平均 SMAPE)
# MAGIC
# MAGIC まず、すべての系列と backtest window を平均した「全体ランキング」を見る。
# MAGIC これが「どのモデルが平均的に強いか」の表。
# MAGIC
# MAGIC `actual` / `forecast` は配列なので、`zip_with` + `aggregate` で配列のまま SMAPE を計算する。

# COMMAND ----------

ranking_sql = f"""
WITH smape_per_window AS (
  SELECT
    unique_id,
    model,
    aggregate(
      zip_with(actual, forecast, (a, f) ->
        CASE WHEN abs(a) + abs(f) = 0 THEN 0.0
             ELSE 2.0 * abs(a - f) / (abs(a) + abs(f)) END
      ),
      CAST(0.0 AS DOUBLE),
      (acc, x) -> acc + x
    ) / size(actual) AS window_smape
  FROM {CATALOG}.{SCHEMA}.{EVAL_TABLE}
  WHERE use_case = '{USE_CASE}'
)
SELECT
  model,
  ROUND(AVG(window_smape), 4) AS avg_smape,
  COUNT(DISTINCT unique_id) AS n_series,
  COUNT(*) AS n_windows
FROM smape_per_window
GROUP BY model
ORDER BY avg_smape ASC
"""
display(spark.sql(ranking_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 読み方
# MAGIC
# MAGIC `avg_smape` が一番小さいモデルが「全体 1 位」。
# MAGIC ここで「じゃあ 1 位のモデルを全系列に使えばいいのでは?」と思うのが自然な反応。
# MAGIC 次のステップでその直感がひっくり返る。

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ8: 系列ごとのチャンピオンモデル
# MAGIC
# MAGIC MMF の核心。「全体 1 位のモデル ≠ ある系列での 1 位」ということが、5 系列でもしばしば観察できる。

# COMMAND ----------

champion_sql = f"""
WITH smape_per_window AS (
  SELECT
    unique_id, model,
    aggregate(
      zip_with(actual, forecast, (a, f) ->
        CASE WHEN abs(a) + abs(f) = 0 THEN 0.0
             ELSE 2.0 * abs(a - f) / (abs(a) + abs(f)) END
      ),
      CAST(0.0 AS DOUBLE),
      (acc, x) -> acc + x
    ) / size(actual) AS window_smape
  FROM {CATALOG}.{SCHEMA}.{EVAL_TABLE}
  WHERE use_case = '{USE_CASE}'
),
per_series AS (
  SELECT unique_id, model, AVG(window_smape) AS smape
  FROM smape_per_window
  GROUP BY unique_id, model
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY smape) AS rnk
  FROM per_series
)
SELECT unique_id, model AS champion_model, ROUND(smape, 4) AS smape
FROM ranked
WHERE rnk = 1
ORDER BY unique_id
"""
display(spark.sql(champion_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 読み方
# MAGIC
# MAGIC `champion_model` 列を縦に眺めて、**全系列で同じモデルが並んでいなければ MMF の意味あり**。
# MAGIC 5 系列という小さなサンプルでも、複数モデルがチャンピオンに顔を出すことが多い。
# MAGIC
# MAGIC 実運用 (数千〜数百万系列) では、ここでチャンピオンが大きく分散する。
# MAGIC 「全系列に同じモデル」で予測すると、その分の精度を取りこぼしている、というのが MMF の価値提案。

# COMMAND ----------

# MAGIC %md
# MAGIC ## チャンピオン回数を棒グラフで可視化
# MAGIC
# MAGIC 5 系列中、各モデルが何系列でチャンピオンになったかを集計してプロットする。
# MAGIC 1 つのモデルが独占しているか、分散しているかが一目で分かる。

# COMMAND ----------

champion_count_pdf = spark.sql(f"""
WITH smape_per_window AS (
  SELECT
    unique_id, model,
    aggregate(
      zip_with(actual, forecast, (a, f) ->
        CASE WHEN abs(a) + abs(f) = 0 THEN 0.0
             ELSE 2.0 * abs(a - f) / (abs(a) + abs(f)) END
      ),
      CAST(0.0 AS DOUBLE),
      (acc, x) -> acc + x
    ) / size(actual) AS window_smape
  FROM {CATALOG}.{SCHEMA}.{EVAL_TABLE}
  WHERE use_case = '{USE_CASE}'
),
per_series AS (
  SELECT unique_id, model, AVG(window_smape) AS smape
  FROM smape_per_window
  GROUP BY unique_id, model
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY smape) AS rnk
  FROM per_series
)
SELECT model, COUNT(*) AS n_series_where_best
FROM ranked
WHERE rnk = 1
GROUP BY model
ORDER BY n_series_where_best DESC
""").toPandas()

fig = px.bar(
    champion_count_pdf,
    x="model",
    y="n_series_where_best",
    title=f"チャンピオン回数 ({N_SERIES} 系列中、何系列で 1 位か)",
    text="n_series_where_best",
)
fig.update_layout(xaxis_title="", yaxis_title="チャンピオン回数", height=400)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ9: 1系列のバックテストを可視化
# MAGIC
# MAGIC 「全体ランキング」「チャンピオン回数」だけでは、モデルごとの予測の癖までは見えない。
# MAGIC 1 系列を選んで、最終バックテスト window で実測と各モデルの予測を重ねてみる。
# MAGIC
# MAGIC それぞれのモデルの「外し方」が直感的に分かる。

# COMMAND ----------

focus_uid = target_ids[0]

bt_long = spark.sql(f"""
WITH base AS (
  SELECT
    unique_id,
    model,
    backtest_window_start_date,
    posexplode(arrays_zip(actual, forecast)) AS (idx, vals)
  FROM {CATALOG}.{SCHEMA}.{EVAL_TABLE}
  WHERE use_case = '{USE_CASE}'
    AND unique_id = '{focus_uid}'
),
latest_window AS (
  SELECT MAX(backtest_window_start_date) AS d FROM base
)
SELECT
  base.model,
  date_add(CAST(base.backtest_window_start_date AS DATE), base.idx) AS ds,
  base.vals.actual AS actual,
  base.vals.forecast AS forecast
FROM base
JOIN latest_window ON base.backtest_window_start_date = latest_window.d
ORDER BY base.model, ds
""").toPandas()

# COMMAND ----------

import plotly.graph_objects as go

fig = go.Figure()

# 実測 (どのモデルでも同じ actual なので 1 本だけ太く描く)
actual_pdf = bt_long.drop_duplicates(subset=['ds']).sort_values('ds')
fig.add_trace(go.Scatter(
    x=actual_pdf['ds'], y=actual_pdf['actual'],
    mode='lines+markers', name='actual',
    line=dict(color='black', width=4),
    marker=dict(size=8),
))

# 各モデルの予測
for model_name in sorted(bt_long['model'].unique()):
    pdf_m = bt_long[bt_long['model'] == model_name].sort_values('ds')
    fig.add_trace(go.Scatter(
        x=pdf_m['ds'], y=pdf_m['forecast'],
        mode='lines+markers', name=model_name,
        line=dict(width=2, dash='dot'),
        marker=dict(size=6),
    ))

fig.update_layout(
    title=f"{focus_uid} の最終バックテスト window: 実測 vs 各モデル予測",
    xaxis_title="日付",
    yaxis_title="y",
    height=500,
    hovermode='x unified',
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 読み方
# MAGIC
# MAGIC 黒線が実測、点線が各モデルの予測。
# MAGIC
# MAGIC - 直近の傾きをそのまま延長するモデル (AutoArima、AutoTheta) と
# MAGIC - 水平に予測するモデル (Naive) と
# MAGIC - 7 日前の値を繰り返すモデル (SeasonalNaive)
# MAGIC
# MAGIC で予測の形がまったく違うことが見える。SMAPE という 1 つの数字に圧縮される前の
# MAGIC 「モデルごとの個性」がここに出ている。
# MAGIC
# MAGIC `focus_uid` を `target_ids[1]`、`target_ids[2]` ... に変えて他系列でも試すと、
# MAGIC 同じモデルでも系列によって振る舞いが違うことが見て取れる。

# COMMAND ----------

# MAGIC %md
# MAGIC # まとめ
# MAGIC
# MAGIC - リポジトリ clone 不要。`pip install "mmf_sa[local] @ git+https://..."` だけで MMF を導入。
# MAGIC - 公式 `mmf_sa.run_forecast` をそのまま呼んでいるので、`evaluation_output` / `scoring_output` の
# MAGIC   構造は本家と同じ。`actual` / `forecast` は配列型 (長さ 10)。
# MAGIC - 縮小したのは入力規模 (系列数・履歴長・モデル数・window 数) のみで、MMF のコア体験は維持。
# MAGIC - 全体ランキング → 系列ごとのチャンピオン → 1系列のバックテスト可視化、という 3 段階で
# MAGIC   「系列ごとに最適モデルが違う」を体感する流れにしてある。
# MAGIC - 後処理 SQL を本格的に試したい場合は元の記事
# MAGIC   (https://qiita.com/taka_yayoi/items/0ec31085025ffde092e5) のクエリ群を参照。
