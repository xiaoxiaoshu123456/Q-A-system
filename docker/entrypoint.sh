#!/bin/sh
# Entrypoint for containerized app:
# - Optionally override config.ini values via env vars (Milvus + LLM).
# - Launch Streamlit web UI.
set -e
python - <<'PY'
import os,configparser
p="config.ini"
parser=configparser.ConfigParser()
parser.read(p,encoding="utf-8")
if not parser.has_section("milvus"):
    parser.add_section("milvus")
host=os.environ.get("MILVUS_HOST") or parser.get("milvus","host",fallback="milvus")
port=os.environ.get("MILVUS_PORT") or parser.get("milvus","port",fallback="19530")
parser.set("milvus","host",str(host))
parser.set("milvus","port",str(port))
if not parser.has_section("llm"):
    parser.add_section("llm")
prov=os.environ.get("LLM_PROVIDER")
model=os.environ.get("LLM_MODEL")
api_key=os.environ.get("LLM_API_KEY")
base_url=os.environ.get("LLM_BASE_URL")
if prov: parser.set("llm","provider",prov)
if model: parser.set("llm","model",model)
if api_key: parser.set("llm","api_key",api_key)
if base_url: parser.set("llm","base_url",base_url)
with open(p,"w",encoding="utf-8") as f:
    parser.write(f)
PY
exec streamlit run web_ui.py --server.port "${PORT:-8501}" --server.headless true
