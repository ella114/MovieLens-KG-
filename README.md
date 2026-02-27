# MovieLens + 知识图谱推荐系统（最小可运行骨架）

本项目提供一个可复现实验管线，对比：

- 传统协同过滤：`ItemCF`
- 仅行为矩阵分解：`SVD`
- 融合电影知识图谱：`KG-GCN`（类别/导演/演员关系图）

## 1. 目录结构

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── scripts/
│   └── run_all.sh
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── build_kg.py
│   ├── fetch_tmdb_metadata.py
│   ├── train_cf.py
│   ├── train_svd.py
│   ├── train_kg.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── train_utils.py
│   └── models/
│       ├── cf.py
│       ├── svd.py
│       └── kg_gcn.py
└── requirements.txt
```

## 2. 数据准备

将 MovieLens 文件放到 `data/raw/`，至少包含：

- `ratings.csv`（`userId,movieId,rating,timestamp`）
- `movies.csv`（`movieId,title,genres`）
- 可选：`links.csv`（用于接 TMDB/IMDb）

### 可选：拉取导演/演员元数据（TMDB）

```bash
export TMDB_API_KEY=你的key
python src/fetch_tmdb_metadata.py --raw-dir data/raw --out-file data/raw/tmdb_metadata.csv
```

`preprocess.py` 会自动尝试读取以下文件之一作为外部元数据：

- `data/raw/movie_metadata.csv`
- `data/raw/tmdb_metadata.csv`
- `data/raw/imdb_metadata.csv`
- `data/raw/external_metadata.csv`

要求包含 `movieId` 或 (`imdbId`/`tmdbId`) 以及 `director`、`actors` 列。

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 运行流程

### 一键运行

```bash
bash scripts/run_all.sh data/raw
```

### 分步运行

```bash
python src/preprocess.py --raw-dir data/raw
python src/build_kg.py
python src/train_cf.py
python src/train_svd.py
python src/train_kg.py
python src/evaluate.py
python src/visualize.py --model-dir kg_gcn
```

## 5. 默认超参数

- `SVD`
  - `embedding_dim=64`
  - `lr=1e-3`
  - `weight_decay=1e-4`
  - `batch_size=4096`
  - `epochs=50`
  - `early_stop=5`

- `KG-GCN`
  - `hidden_dim=128`
  - `num_layers=2`
  - `dropout=0.2`
  - `lr=2e-3`
  - `weight_decay=1e-4`
  - `batch_size=2048`
  - `epochs=80`
  - `early_stop=8`
  - `alpha=0.2`（BPR 辅助损失权重）

- `KG 构图`
  - 同导演边：`w_director=1.0`
  - 同演员边：`w_actor=0.7`, 最少重叠演员数 `min_actor_overlap=2`
  - 同类别边：`w_genre=0.5`, `genre_jaccard>=0.5`

## 6. 输出结果

- `outputs/cf/metrics.json`
- `outputs/svd/metrics.json`
- `outputs/kg_gcn/metrics.json`
- `outputs/metrics_comparison.csv`
- `outputs/figures/metrics_comparison.png`
- `outputs/kg_gcn/top10_with_reasons_user*.csv`

## 7. 评估指标

- 显式评分：`RMSE`
- Top-K：`Precision@K`、`Recall@K`、`F1@K`

其中正反馈阈值默认 `rating >= 4.0`。

## 8. 备注

- 当前 `KG-GCN` 使用“电影-电影同构图”与电影侧特征联合训练。
- 如果你需要严格异构图（movie/director/actor/genre 多类型节点），可扩展为 `R-GCN/HeteroGNN`。
