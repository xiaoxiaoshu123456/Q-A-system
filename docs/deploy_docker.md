# Docker 部署

## 方案 A：仅启动 Milvus
```
pwsh docker/up_milvus.ps1
```
Milvus 监听 19530，本机应用可直接使用该端口，容器内连接地址为 milvus-standalone。

## 方案 B：同时启动 Milvus 与应用
```
pwsh docker/up_all.ps1
```
默认端口：
- 应用：8501
- Milvus：19530
- Etcd：2379
- MinIO：9000/9001

## 手动构建应用镜像
```
pwsh docker/build_app.ps1
```
容器内支持环境变量覆盖配置：
- MILVUS_HOST、MILVUS_PORT
- LLM_PROVIDER、LLM_MODEL、LLM_API_KEY、LLM_BASE_URL

