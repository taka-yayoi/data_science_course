# データサイエンス講義

Databricks Free Editionを用いた機械学習・MLOpsハンズオン教材

## 概要

本リポジトリは、大学院生向けのデータサイエンス講義で使用するノートブック教材です。

## 前提条件

- Databricks Free Editionアカウント
- データエンジニアリング編(Spark、Delta Lake、パイプライン)の知識

## フォルダ構成

```
ds-workshop/
├── README.md
└── demos/                              # デモ用ノートブック(講師用)
    ├── 01_ml_environment_check.py      # Databricks ML環境の確認
    ├── 02_sklearn_pipeline.py          # scikit-learnパイプラインの構築
    ├── 03_mlflow_tracking.py           # MLflowによる実験トラッキング
    └── 04_model_registry.py            # UCモデルレジストリとChampion/Challengerパターン
```

## 講義構成

### Part 1: ノートブックによるモデル開発実践 (90分)
- 機械学習の基本概念
- scikit-learnによるモデル開発
- MLflowによる実験管理

### Part 2: MLOpsによる業務品質のモデル開発サイクル (90分)
- なぜMLOpsが必要か
- 這う・歩く・走るアプローチ
- Unity Catalog / MLflow / Model Registry

### Part 3: 実践演習 (90分)
- scikit-learn + MLflowで機械学習ワークフロー
- ※実習用ノートブックは別途作成予定

## 使用方法

### Gitリポジトリからのインポート

1. Databricksワークスペースにログイン
2. 左サイドバーの「ワークスペース」をクリック
3. 「Gitフォルダを追加」を選択
4. リポジトリURLを入力してインポート

### 計算リソースについて

Free Editionではサーバーレスコンピュートを使用します。

**注意事項**:
- 複数ノートブックを同時に実行するとエラーが発生する場合があります
- 1つのノートブックの実行が完了したら、右上の「接続済み」→「終了」で接続を切断してから次のノートブックを実行してください
- SparkML(pyspark.ml)はサーバーレス環境では利用できないため、scikit-learnを使用しています

## 使用データセット

- Wine Quality Dataset (UCI Machine Learning Repository)
  - Databricksサンプルデータセットとして利用可能
  - 赤ワインの品質を予測する分類タスク
  - パス: `/databricks-datasets/wine-quality/winequality-red.csv`

## 技術スタック

| 用途 | 技術 |
|------|------|
| データI/O | Spark DataFrame, Delta Lake |
| 機械学習 | scikit-learn |
| 実験管理 | MLflow Tracking |
| モデル管理 | MLflow Model Registry (Unity Catalog) |

## ライセンス

教育目的での使用を想定しています。
