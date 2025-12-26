# 架构设计文档

## 概览
- 技术栈：Python、Streamlit、Milvus、BGE-M3、OpenAI 兼容 API、Anthropic
- 模块划分：
  - Web 界面：[web_ui.py](../web_ui.py)
  - 数据库与检索：[init_database.py](../init_database.py)、[vector_store.py](../vector_store.py)
  - 入库记录与哈希：[upload_manifest.py](../upload_manifest.py)
  - 对话日志留存：[conversation_log.py](../conversation_log.py)
  - 配置文件：[config.ini](../config.ini)

## 数据流程
- 上传入库：文件上传 → 文本分块 → 稠密/稀疏向量生成 → 写入 Milvus → 记录到 logs/uploads.json
- 查询对话：问题向量化 → 混合检索 → 父块去重聚合 → 调用 LLM → 展示答案 → 写入 logs/conversations.jsonl
- 同步历史：从 Milvus 扫描 grandpa_id → 合并到 uploads.json（名称未知时可自动命名）

## 关键设计
- 混合检索：BGE-M3 稠密向量 + 稀疏向量加权融合
- 多轮对话：保留最近 5 轮上下文，构建 LLM 上下文
- LLM Provider：OpenAI 兼容、Claude、Qwen（DashScope）、Ollama、vLLM，统一在侧边栏配置
- 日志与留存：运行日志 logs/app.log；问答留存 logs/conversations.jsonl
- 文件哈希与缓存：固定路径 logs/upload_cache/{hash}.{ext}，避免重复入库

## 配置说明
- [config.ini](../config.ini)
  - [milvus]：host、port、database_name、collection_name
  - [llm]：provider、model、api_key、base_url（或 dashscope/anthropic 字段）
  - [retrieval]：parent_chunk_size、child_chunk_size、chunk_overlap、retrieval_k、candidate_m

## 重要函数与位置
- UI 对话日志写入：[web_ui.py](../web_ui.py#L296-L320)
- Provider 分发与调用：[init_database.py](../init_database.py#L624-L686)
- 入库与删除封装：[vector_store.py](../vector_store.py#L148-L190)
- 入库记录读写：[upload_manifest.py](../upload_manifest.py)
- 自动命名来源：[web_ui.py](../web_ui.py#L90-L104)

## 安全与运维
- 密钥仅存储在 config.ini 的 llm 段
- app.log 不记录密钥
- conversations.jsonl 用于离线分析，不包含敏感密钥

## 其他文档
-
- 

