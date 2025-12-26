# Docker 和 Milvus 向量数据库安装指南

## 1. Docker 安装

### Linux 系统（Ubuntu/Debian）

bash

```
# 1. 更新软件包索引
sudo apt-get update

# 2. 安装依赖包
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 3. 添加 Docker 官方 GPG 密钥
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. 设置存储库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. 更新包索引并安装 Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 6. 验证安装
sudo docker run hello-world

# 7. （可选）添加当前用户到 docker 组，避免每次使用 sudo
sudo usermod -aG docker $USER
# 需要注销重新登录生效
```

### macOS 系统

1. 访问 [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. 下载安装包（Intel 芯片或 Apple Silicon）
3. 双击下载的 .dmg 文件
4. 将 Docker 图标拖到 Applications 文件夹
5. 启动 Docker Desktop
6. 在终端验证：`docker --version`

### Windows 系统

1. 访问 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. 下载安装包
3. 运行安装程序（需要启用 WSL2 或 Hyper-V）
4. 按照提示重启计算机
5. 启动 Docker Desktop
6. 在 PowerShell 或 CMD 中验证：`docker --version`

## 2. Milvus 安装

### 方法一：使用 Docker Compose（单机版）

bash

```
# 1. 下载 docker-compose.yml 文件
wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 2. 启动 Milvus
docker-compose up -d

# 3. 检查运行状态
docker-compose ps

# 4. 停止 Milvus
docker-compose down
```

### 方法二：使用 Docker 直接运行

bash

```
# 1. 拉取 Milvus 镜像
docker pull milvusdb/milvus:v2.3.3

# 2. 创建数据存储目录
mkdir -p /path/to/milvus/data
mkdir -p /path/to/milvus/configs

# 3. 运行 Milvus 容器
docker run -d \
    --name milvus-standalone \
    -p 19530:19530 \
    -p 9091:9091 \
    -v /path/to/milvus/data:/var/lib/milvus \
    -v /path/to/milvus/configs:/etc/milvus \
    milvusdb/milvus:v2.3.3
```

### 方法三：使用 Helm 安装（Kubernetes）

bash

```
# 1. 添加 Milvus Helm 仓库
helm repo add milvus https://milvus-io.github.io/milvus-helm/

# 2. 更新 Helm 仓库
helm repo update

# 3. 安装 Milvus
helm install my-milvus milvus/milvus --set cluster.enabled=false
```

## 3. 配置 Milvus

### 创建配置文件（可选）

yaml

```
# configs/standalone.yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

## 4. 验证安装

### 检查服务状态

bash

```
# 检查 Docker 容器运行状态
docker ps

# 检查 Milvus 日志
docker logs milvus-standalone

# 使用 curl 检查 Milvus 健康状态
curl http://localhost:9091/healthz

# 或者使用 milvus-cli 工具
docker run --rm -it milvusdb/milvus:v2.3.3 milvus-cli --host localhost --port 19530
```

### 使用 Python 客户端测试连接

python

```
# 安装 Python 客户端
pip install pymilvus

# 测试连接
from pymilvus import connections, utility

# 连接 Milvus
connections.connect(
    alias="default", 
    host='localhost', 
    port='19530'
)

# 检查连接
print(utility.get_server_version())
print(utility.list_collections())
```

## 5. 常用命令

### Docker 常用命令

bash

```
# 查看 Docker 版本
docker --version
docker-compose --version

# 查看运行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 停止容器
docker stop <container_id>

# 启动容器
docker start <container_id>

# 删除容器
docker rm <container_id>

# 查看日志
docker logs <container_id>

# 进入容器
docker exec -it <container_id> /bin/bash
```

### Milvus 管理命令

bash

```
# 重启 Milvus
docker-compose restart

# 查看 Milvus 日志
docker-compose logs -f standalone

# 备份数据
docker exec milvus-standalone milvus-backup backup -n backup_name

# 恢复数据
docker exec milvus-standalone milvus-backup restore -n backup_name
```

## 6. 故障排除

### 常见问题

1. **端口冲突**

   bash

   ```
   # 检查端口占用
   sudo netstat -tulpn | grep :19530
   # 或修改 docker-compose.yml 中的端口映射
   ```

2. **权限问题**

   bash

   ```
   # 确保用户有 Docker 权限
   sudo usermod -aG docker $USER
   
   # 确保数据目录有正确权限
   sudo chmod -R 777 /path/to/milvus/data
   ```

3. **内存不足**

   bash

   ```
   # 查看 Docker 资源使用
   docker stats
   
   # 增加 Docker 内存限制（Docker Desktop 设置中）
   ```

4. **连接失败**

   bash

   ```
   # 检查防火墙
   sudo ufw status
   sudo ufw allow 19530/tcp
   sudo ufw allow 9091/tcp
   ```

## 7. 生产环境建议

1. **使用 Docker Swarm 或 Kubernetes** 部署集群版
2. **配置数据持久化** 到外部存储
3. **设置资源限制**（CPU、内存）
4. **启用监控**（Prometheus + Grafana）
5. **定期备份**数据
6. **使用网络隔离**（自定义 Docker 网络）

## 8. 卸载

### 卸载 Milvus

bash

```
# 停止并删除容器
docker-compose down -v

# 或直接删除容器
docker stop milvus-standalone
docker rm milvus-standalone
```

### 卸载 Docker

bash

```
# Ubuntu/Debian
sudo apt-get purge docker-ce docker-ce-cli containerd.io
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd

# macOS/Windows: 通过 Docker Desktop 卸载程序
```

这个指南涵盖了 Docker 和 Milvus 的基本安装步骤。根据你的具体需求和环境，可能需要进行适当的调整。