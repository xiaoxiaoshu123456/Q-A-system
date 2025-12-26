# 界面操作手册（Streamlit 前端）

## 概览
- 访问地址：`http://localhost:8501`
- 页面结构：侧边栏设置 + 顶部标签页（查询、上传）
- 主要用途：检索入库文档并进行问答、管理入库记录、配置模型服务商

## 侧边栏设置
- 调用大模型生成答案：开启后，答案在知识库上下文基础上调用 LLM 生成（默认开启）
- 客服电话：用于“信息不足”兜底文案中的电话
- LLM 配置（支持 OpenAI、Claude、Qwen、Ollama、本地 vLLM、自定义）
  - 预设模型与一键应用：可快速写入 provider、model、base_url、api_key
  - 修改后点击“应用配置”生效，并写入配置文件
  - 代码位置参考：[web_ui.py](../web_ui.py#L170-L221)
- 检索参数
  - top_k：返回文档块数量
  - 稠密权重、稀疏权重：两种向量融合权重
- 入库参数
  - batch_size：分批写入 Milvus 的批大小

## 查询页（标签：查询）
- 多轮对话
  - 保留最近 5 轮上下文，影响 LLM 的回答（自动裁剪，越聊越稳）
  - 可点击“清空对话”重置
  - 代码位置：[web_ui.py](../web_ui.py#L232-L320)
- 提问与回答
  - 在底部输入问题，回车发送
  - 系统会进行“向量化 + 混合检索”，得到父块内容后拼接为上下文
  - 若开启 LLM，会在上下文基础上调用模型生成最终答案
  - 回答为空或失败时，显示“信息不足”兜底文案
- 对话留存与耗时
  - 每轮对话会写入 `logs/conversations.jsonl`
  - 字段包含：问题、答案、provider、model、base_url、检索命中统计，以及 `retrieval_ms`、`llm_ms`、`response_ms`
  - 写入位置：[web_ui.py](../web_ui.py#L296-L320)

## 上传页（标签：上传）
- 选择文件并入库
  - 支持 txt / md / pdf / docx
  - 点击“开始入库”后，文件会被分块、生成稠密/稀疏向量并写入 Milvus
  - 入库成功后，记录插入 `logs/uploads.json`
  - 重复文件（哈希一致）默认跳过并提示
  - 入库主流程位置：[web_ui.py](../web_ui.py#L322-L410)，封装类：[vector_store.py](../vector_store.py#L148-L190)
- 清空结果
  - 清空本地入库记录（不会影响 Milvus）
- 同步已入库文档
  - 从 Milvus 扫描 grandpa_id，将历史入库补齐到本地记录
  - 新增条目以“(未知)”作为文件名占位
  - 代码位置：[web_ui.py](../web_ui.py#L411-L439)
- 自动命名与保存
  - 对“(未知)”条目，可“自动命名并保存”从 Milvus抓取该文档首行作为标题
  - 也可手动输入并保存名称（持久化到 `logs/uploads.json`）
  - 代码位置：[web_ui.py](../web_ui.py#L90-L104) 与命名按钮逻辑 [web_ui.py](../web_ui.py#L289-L313)

## 运行日志与文件
- 运行日志：`logs/app.log`（包含启动与检索过程状态）
- 入库记录：`logs/uploads.json`（“文件/状态/grandpa_id/时间/hash”）
- 对话留存：`logs/conversations.jsonl`（每行一个 JSON）

## 常见操作步骤
- 配置模型服务商
  - 在侧边栏“LLM 配置”选择服务商或点“一键应用”，填入 API Key，应用配置
  - Qwen（DashScope）推荐：`model=qwen-plus`、`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`
- 首次入库
  - 上传文档 → 开始入库 → 成功后在“已入库文件”列表看到记录
- 历史同步
  - 打开“同步已入库文档”，配置数量 → 点击“从 Milvus 同步”
  - 对“(未知)”条目执行“自动命名并保存”
- 提问对话
  - 在“查询”页输入问题 → 观察检索与答案 → 刷新后记录可在 `conversations.jsonl` 中查看

## 故障与提示
- 信息不足：上下文不足或模型不可用时会显示兜底提示，可检查 LLM 配置或提升检索权重
- 已入库但未显示名称：“自动命名并保存”或手动保存名称即可持久化到 `uploads.json`
- 连接 Milvus 失败：检查 `config.ini` 的 `milvus.host/port` 或使用 Docker 一体化部署

