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
├── demos/                              # デモ用ノートブック(講師用)
│   ├── 01_ml_environment_check.py      # Databricks ML環境の確認
│   ├── 02_sparkml_pipeline.py          # SparkMLパイプラインの構築
│   ├── 03_mlflow_tracking.py           # MLflowによる実験トラッキング
│   ├── 04_model_registry.py            # モデルレジストリの操作
│   └── 05_batch_inference.py           # バッチ推論
├── exercises/                          # 実習用ノートブック(受講者用)
│   ├── exercise_sparkml_mlflow.py      # SparkML+MLflow実習(穴埋め形式)
│   └── solutions/
│       └── exercise_sparkml_mlflow_solution.py  # 解答
└── includes/
    └── setup.py                        # 共通設定
```

## 講義構成

### Part 1: ノートブックによるモデル開発実践 (90分)
- 機械学習の基本概念
- SparkMLによるモデル開発
- MLflowによる実験管理

### Part 2: MLOpsによる業務品質のモデル開発サイクル (90分)
- なぜMLOpsが必要か
- 這う・歩く・走るアプローチ
- Unity Catalog / MLflow / Feature Store

### Part 3: 実践演習 (90分)
- SparkML + MLflowで機械学習ワークフロー

## 使用方法

### Gitリポジトリからのインポート

1. Databricksワークスペースにログイン
2. 左サイドバーの「ワークスペース」をクリック
3. 「Gitフォルダを追加」を選択
4. リポジトリURLを入力してインポート

### 計算リソースについて

Free Editionではサーバーレスコンピュートを使用します。

**注意**: 複数ノートブックを同時に実行するとエラーが発生する場合があります。
1つのノートブックの実行が完了したら、右上の「接続済み」→「終了」で接続を切断してから次のノートブックを実行してください。

## 使用データセット

- Wine Quality Dataset (UCI Machine Learning Repository)
  - Databricksサンプルデータセットとして利用可能
  - 赤ワインの品質を予測する分類/回帰タスク

## ライセンス

教育目的での使用を想定しています。
