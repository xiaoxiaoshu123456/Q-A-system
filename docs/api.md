# 接口文档

## 配置
- 文件：[config.ini](../config.ini)
- 段：
  - [milvus]：`host`、`port`、`database_name`、`collection_name`
  - [llm]：`provider`、`model`、`api_key`、`base_url`（或 `dashscope_*`、`anthropic_*`）
  - [retrieval]：`parent_chunk_size`、`child_chunk_size`、`chunk_overlap`、`retrieval_k`、`candidate_m`

## CLI
- 入口：[init_database.py](../init_database.py)
- 示例：
  - 初始化入库：`python init_database.py --docs docs/file1.md docs/file2.txt --batch-size 16`
  - 询问问题：`python init_database.py --question "xxx" --top-k 20 --dense-weight 1.0 --sparse-weight 0.8`
  - 删除文档：`python init_database.py --delete-grandpa-id <gid>`
  - 使用自定义模型地址：`python init_database.py --config config.ini --model-path model/bge-m3`

## 对话留存
- 文件：`logs/conversations.jsonl`
- 单行 JSON 字段：
  - `trace_id`、`question`、`answer`
  - `provider`、`model`、`base_url`
  - `top_k`、`dense_weight`、`sparse_weight`
  - `parents`、`hits`
  - `retrieval_ms`、`llm_ms`、`response_ms`
  - `ts`

## 入库记录
- 文件：`logs/uploads.json`
- 字段：
  - `文件`、`状态`、`grandpa_id`、`时间`、`hash`

