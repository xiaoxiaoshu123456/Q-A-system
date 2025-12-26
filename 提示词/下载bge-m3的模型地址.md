#由于bgem3大于2G,上传不上去,所以需要自己下载

from huggingface_hub import snapshot_download

# 下载整个模型
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="./bge-m3-model",
    local_dir_use_symlinks=False
)

# 或只下载特定文件
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="BAAI/bge-m3",
    filename="pytorch_model.bin",
    local_dir="./bge-m3-model"
)
