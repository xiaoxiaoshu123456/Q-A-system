"""Web UI for the document QA system.

- Provides two tabs: 查询 (chat/QA) and 上传 (ingest).
- Sidebar contains LLM provider configuration, retrieval and ingest params.
- Persists upload records to logs/uploads.json and chat logs to logs/conversations.jsonl.
- Integrates Milvus for hybrid retrieval (dense + sparse) and multiple LLM providers.
"""
import json
import time
from pathlib import Path

import streamlit as st
from pymilvus import Collection, utility

from init_database import (
    _call_llm_from_config,
    _connect_milvus,
    _embed_query,
    _hybrid_search,
    _parse_config,
    _unique_parents,
    get_app_logger,
)
from upload_manifest import find_by_hash, load_manifest, md5_hex, save_manifest, update_manifest_name
from conversation_log import append as append_conv_log
from vector_store import VectorStore


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _upload_cache_dir(root: Path) -> Path:
    return root / "logs" / "upload_cache"


def _new_trace_id() -> str:
    return hex(time.time_ns())[2:]


def _format_history(messages: list[dict], max_messages: int = 8) -> str:
    if not messages:
        return ""
    max_messages = max(0, int(max_messages))
    selected = messages[-max_messages:] if max_messages else []
    lines: list[str] = []
    for m in selected:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
        else:
            lines.append(content)
    return "\n".join(lines).strip()


def _build_llm_context(history_text: str, kb_context: str) -> str:
    history_text = (history_text or "").strip()
    kb_context = (kb_context or "").strip()
    if history_text and kb_context:
        return f"对话历史：\n{history_text}\n\n知识库内容：\n{kb_context}"
    if history_text:
        return f"对话历史：\n{history_text}"
    return kb_context


def _manifest_path(root: Path) -> Path:
    return root / "logs" / "uploads.json"


def _load_manifest(root: Path) -> list[dict]:
    return load_manifest(root)


def _save_manifest(root: Path, items: list[dict]) -> None:
    save_manifest(root, items)


def _update_manifest_name(root: Path, grandpa_id: str, new_name: str) -> None:
    update_manifest_name(root, grandpa_id, new_name)


def _scan_grandpa_ids_from_milvus(config_path: Path, max_unique: int = 50, scan_rows: int = 3000, page_size: int = 300) -> list[str]:
    col = _collection_from_ini(config_path)
    expr = "grandpa_id != ''"
    out: list[str] = []
    seen: set[str] = set()
    offset = 0
    max_unique = max(0, int(max_unique))
    scan_rows = max(0, int(scan_rows))
    page_size = max(1, int(page_size))
    while offset < scan_rows and len(out) < max_unique:
        limit = min(page_size, scan_rows - offset)
        rows = col.query(expr=expr, output_fields=["grandpa_id"], offset=offset, limit=limit)
        if not rows:
            break
        for r in rows:
            gid = (r.get("grandpa_id") or "").strip()
            if not gid or gid in seen:
                continue
            seen.add(gid)
            out.append(gid)
            if len(out) >= max_unique:
                break
        offset += len(rows)
    return out


def _suggest_name_from_milvus(config_path: Path, grandpa_id: str, max_len: int = 40) -> str:
    col = _collection_from_ini(config_path)
    rows = col.query(
        expr=f"grandpa_id == '{(grandpa_id or '').strip()}'",
        output_fields=["parend_context", "text"],
        limit=1,
    )
    if not rows:
        return ""
    ctx = (rows[0].get("parend_context") or rows[0].get("text") or "").strip()
    title = ctx.splitlines()[0].strip() if ctx else ""
    title = title[:max_len]
    return title

def _collection_from_ini(config_path: Path) -> Collection:
    milvus_cfg, _ = _parse_config(config_path)
    _connect_milvus(milvus_cfg)
    if not utility.has_collection(milvus_cfg.collection_name):
        raise RuntimeError(f"collection 不存在: {milvus_cfg.collection_name}")
    col = Collection(milvus_cfg.collection_name)
    col.load()
    return col


st.set_page_config(page_title="文档问答系统", layout="wide")
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)
st.title("文档问答系统")

root = _project_root()
config_path = root / "config.ini"
model_path = root / "model" / "bge-m3"
logger = get_app_logger()

if not config_path.exists():
    st.error(f"找不到配置文件: {config_path}")
    st.stop()
if not model_path.exists():
    st.error(f"找不到模型目录: {model_path}")
    st.stop()

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "uploads_manifest" not in st.session_state:
    st.session_state["uploads_manifest"] = _load_manifest(root)
if not st.session_state.get("_logged_start"):
    logger.info("web_ui start")
    st.session_state["_logged_start"] = True

with st.sidebar:
    st.subheader("设置")
    use_llm = st.toggle("调用大模型生成答案", value=True)
    phone = st.text_input("客服电话", value="10086")
    with st.expander("LLM 配置", expanded=False):
        provider_ui = st.selectbox(
            "服务商",
            options=["OpenAI", "Claude", "Qwen (DashScope 兼容)", "Ollama（本地）", "vLLM（本地/自建）", "自定义"],
            index=2,
        )
        provider = "anthropic" if provider_ui == "Claude" else "openai_compat"
        preset = st.selectbox("预设模型", options=["qwen-plus", "自定义"], index=0)
        llm_model = st.text_input("模型", value=("qwen-plus" if preset == "qwen-plus" else ""))
        base_url_in = st.text_input("Base URL", value=st.session_state.get("llm_base_url", ""))
        api_key_in = st.text_input("API Key", value=st.session_state.get("llm_api_key", ""), type="password")
        apply_llm = st.button("应用配置", use_container_width=True)
        apply_qwen = st.button("一键应用 Qwen-Plus（DashScope）", use_container_width=True)
        apply_openai = st.button("一键应用 OpenAI", use_container_width=True)
        apply_ollama = st.button("一键应用 Ollama（本地）", use_container_width=True)
        apply_vllm = st.button("一键应用 vLLM（本地/自建）", use_container_width=True)
        apply_claude = st.button("一键应用 Claude", use_container_width=True)
        if apply_llm:
            try:
                import configparser

                parser = configparser.ConfigParser()
                parser.read(config_path, encoding="utf-8")
                if not parser.has_section("llm"):
                    parser.add_section("llm")
                parser.set("llm", "provider", provider)
                parser.set("llm", "model", llm_model.strip())
                parser.set("llm", "api_key", api_key_in.strip())
                parser.set("llm", "base_url", base_url_in.strip())
                if provider == "anthropic":
                    parser.set("llm", "anthropic_base_url", base_url_in.strip() or "https://api.anthropic.com")
                with open(config_path, "w", encoding="utf-8") as f:
                    parser.write(f)
                st.session_state["llm_base_url"] = base_url_in.strip()
                st.session_state["llm_api_key"] = api_key_in.strip()
                st.success("已应用 LLM 配置")
            except Exception as e:
                st.error(str(e))
        if apply_qwen:
            try:
                import configparser

                parser = configparser.ConfigParser()
                parser.read(config_path, encoding="utf-8")
                if not parser.has_section("llm"):
                    parser.add_section("llm")
                parser.set("llm", "provider", "openai_compat")
                parser.set("llm", "model", "qwen-plus")
                parser.set("llm", "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                if api_key_in.strip():
                    parser.set("llm", "api_key", api_key_in.strip())
                parser.set("llm", "dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                if api_key_in.strip():
                    parser.set("llm", "dashscope_api_key", api_key_in.strip())
                with open(config_path, "w", encoding="utf-8") as f:
                    parser.write(f)
                st.session_state["llm_base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                st.success("已应用 Qwen-Plus（DashScope 兼容）配置")
            except Exception as e:
                st.error(str(e))
        if apply_openai:
            try:
                import configparser

                parser = configparser.ConfigParser()
                parser.read(config_path, encoding="utf-8")
                if not parser.has_section("llm"):
                    parser.add_section("llm")
                parser.set("llm", "provider", "openai_compat")
                parser.set("llm", "model", llm_model.strip() or "gpt-4o-mini")
                parser.set("llm", "base_url", "https://api.openai.com/v1")
                if api_key_in.strip():
                    parser.set("llm", "api_key", api_key_in.strip())
                with open(config_path, "w", encoding="utf-8") as f:
                    parser.write(f)
                st.session_state["llm_base_url"] = "https://api.openai.com/v1"
                st.success("已应用 OpenAI 配置")
            except Exception as e:
                st.error(str(e))
        if apply_ollama:
            try:
                import configparser

                parser = configparser.ConfigParser()
                parser.read(config_path, encoding="utf-8")
                if not parser.has_section("llm"):
                    parser.add_section("llm")
                parser.set("llm", "provider", "openai_compat")
                parser.set("llm", "model", llm_model.strip() or "qwen2:latest")
                parser.set("llm", "base_url", "http://localhost:11434/v1")
                with open(config_path, "w", encoding="utf-8") as f:
                    parser.write(f)
                st.session_state["llm_base_url"] = "http://localhost:11434/v1"
                st.success("已应用 Ollama 配置")
            except Exception as e:
                st.error(str(e))
        if apply_vllm:
            try:
                import configparser

                parser = configparser.ConfigParser()
                parser.read(config_path, encoding="utf-8")
                if not parser.has_section("llm"):
                    parser.add_section("llm")
                parser.set("llm", "provider", "openai_compat")
                parser.set("llm", "model", llm_model.strip())
                parser.set("llm", "base_url", "http://localhost:8000/v1")
                with open(config_path, "w", encoding="utf-8") as f:
                    parser.write(f)
                st.session_state["llm_base_url"] = "http://localhost:8000/v1"
                st.success("已应用 vLLM 配置")
            except Exception as e:
                st.error(str(e))
        if apply_claude:
            try:
                import configparser

                parser = configparser.ConfigParser()
                parser.read(config_path, encoding="utf-8")
                if not parser.has_section("llm"):
                    parser.add_section("llm")
                parser.set("llm", "provider", "anthropic")
                parser.set("llm", "model", llm_model.strip() or "claude-3-5-sonnet-20241022")
                parser.set("llm", "anthropic_base_url", "https://api.anthropic.com")
                if api_key_in.strip():
                    parser.set("llm", "api_key", api_key_in.strip())
                    parser.set("llm", "anthropic_api_key", api_key_in.strip())
                with open(config_path, "w", encoding="utf-8") as f:
                    parser.write(f)
                st.session_state["llm_base_url"] = "https://api.anthropic.com"
                st.success("已应用 Claude 配置")
            except Exception as e:
                st.error(str(e))
    with st.expander("检索参数", expanded=False):
        top_k = st.number_input("top_k", min_value=1, max_value=100, value=20, step=1)
        dense_weight = st.number_input("稠密权重", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        sparse_weight = st.number_input("稀疏权重", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
    with st.expander("入库参数", expanded=False):
        batch_size = st.number_input("batch_size", min_value=1, max_value=256, value=16, step=1)

tab_qa, tab_upload = st.tabs(["查询", "上传"])

with tab_qa:
    max_turns = 5
    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        if st.button("清空对话", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.session_state["last_answer"] = ""
            logger.info("chat cleared")
            st.rerun()
    with col_q2:
        st.caption(f"支持多轮对话：最多保留 {max_turns} 轮上下文。")

    for m in st.session_state["chat_messages"]:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("请输入问题，回车发送…")
    if user_input:
        trace_id = _new_trace_id()
        start_time = time.time()
        user_text = user_input.strip()
        if not user_text:
            st.stop()

        st.session_state["chat_messages"].append({"role": "user", "content": user_text, "ts": time.time()})
        with st.chat_message("user"):
            st.markdown(user_text)

        history_text = _format_history(st.session_state["chat_messages"][:-1], max_messages=max_turns * 2)
        logger.info(
            f"chat turn start trace={trace_id} question={user_text} history_lines={len(history_text.splitlines())} top_k={int(top_k)} dense_w={float(dense_weight)} sparse_w={float(sparse_weight)} use_llm={bool(use_llm)}"
        )

        with st.chat_message("assistant"):
            with st.spinner("向量化 + 混合检索中..."):
                t_retrieval = time.time()
                milvus_cfg, _ = _parse_config(config_path)
                _connect_milvus(milvus_cfg)
                dense_q, sparse_q = _embed_query(model_path, user_text)
                hits = _hybrid_search(
                    milvus_cfg=milvus_cfg,
                    collection_name=milvus_cfg.collection_name,
                    dense_vector=dense_q,
                    sparse_vector=sparse_q,
                    top_k=int(top_k),
                    dense_weight=float(dense_weight),
                    sparse_weight=float(sparse_weight),
                )
                parents = _unique_parents(hits)
                kb_context = "\n\n".join([p.get("parend_context", "") for p in parents if p.get("parend_context")])
                retrieval_ms = int((time.time() - t_retrieval) * 1000)

            answer = kb_context.strip()
            llm_ms = 0
            if use_llm:
                llm_context = _build_llm_context(history_text, kb_context)
                with st.spinner("调用大模型中..."):
                    try:
                        t_llm = time.time()
                        answer = _call_llm_from_config(config_path, user_text, llm_context, phone.strip() or "10086")
                        llm_ms = int((time.time() - t_llm) * 1000)
                    except Exception as e:
                        logger.error(f"chat llm fail trace={trace_id} error={e}")
                        answer = kb_context.strip()

            if not answer:
                answer = f"信息不足，无法回答，请联系人工客服，电话：{phone.strip() or '10086'}。"

            st.markdown(answer)

        st.session_state["chat_messages"].append({"role": "assistant", "content": answer, "ts": time.time()})
        if len(st.session_state["chat_messages"]) > max_turns * 2:
            st.session_state["chat_messages"] = st.session_state["chat_messages"][-max_turns * 2 :]
        st.session_state["last_answer"] = answer
        try:
            import configparser
            parser = configparser.ConfigParser()
            parser.read(config_path, encoding="utf-8")
            prov = parser.get("llm", "provider", fallback="")
            model_used = parser.get("llm", "model", fallback="")
            base_used = (
                parser.get("llm", "base_url", fallback="")
                or parser.get("llm", "dashscope_base_url", fallback="")
                or parser.get("llm", "anthropic_base_url", fallback="")
            )
            response_ms = int((time.time() - start_time) * 1000)
            append_conv_log(
                root,
                {
                    "trace_id": trace_id,
                    "question": user_text,
                    "answer": answer,
                    "provider": prov,
                    "model": model_used,
                    "base_url": base_used,
                    "top_k": int(top_k),
                    "dense_weight": float(dense_weight),
                    "sparse_weight": float(sparse_weight),
                    "parents": len(parents),
                    "hits": len(hits),
                    "retrieval_ms": int(retrieval_ms),
                    "llm_ms": int(llm_ms),
                    "response_ms": int(response_ms),
                },
            )
        except Exception:
            pass
        logger.info(
            f"chat turn done trace={trace_id} hits={len(hits)} parents={len(parents)} answer_chars={len(answer)} kb_context_chars={len(kb_context)} retrieval_ms={int(retrieval_ms)} llm_ms={int(llm_ms)} response_ms={int((time.time()-start_time)*1000)}"
        )

with tab_upload:
    uploaded_files = st.file_uploader(
        "选择要上传的文件（支持 txt / md / pdf / docx）",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    st.caption("提示：选择文件只是本次会话的临时选择，刷新会清空；点击“开始入库”后会在下方“已入库文件”里长期显示。")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        do_ingest = st.button("开始入库", type="primary", use_container_width=True)
    with col_b:
        clear = st.button("清空结果", use_container_width=True)
        if clear:
            st.session_state["uploads_manifest"] = []
            _save_manifest(root, st.session_state["uploads_manifest"])
            logger.info("upload manifest cleared")

    skip_existing = st.toggle("遇到已入库文件：自动跳过并提示", value=True)

    if do_ingest:
        if not uploaded_files:
            st.warning("请先选择文件")
            st.stop()

        logger.info(
            f"ingest start files={len(uploaded_files)} batch_size={int(batch_size)} skip_existing={bool(skip_existing)}"
        )
        with st.spinner("正在入库：向量化可能较慢，请耐心等待..."):
            status = st.status("准备入库", expanded=True)
            vs = VectorStore()
            n = len(uploaded_files)
            ok_count = 0
            skip_count = 0
            fail_count = 0
            for i, f in enumerate(uploaded_files, start=1):
                status.update(label=f"处理中 {i}/{n}: {f.name}", state="running")
                suffix = Path(f.name).suffix
                buf = f.getbuffer()
                size_bytes = len(buf)
                file_hash = md5_hex(bytes(buf))
                existing = find_by_hash(st.session_state["uploads_manifest"], file_hash)
                if existing and skip_existing:
                    logger.info(
                        f"ingest skip existing file={f.name} bytes={size_bytes} hash={file_hash} grandpa_id={existing.get('grandpa_id','')}"
                    )
                    msg = f"检测到 {f.name} 已经入库过了，本次不再重复处理（grandpa_id={existing.get('grandpa_id','')}）。"
                    status.write(msg)
                    st.info(msg)
                    skip_count += 1
                    continue

                cache_dir = _upload_cache_dir(root)
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"{file_hash}{suffix}"
                if not cache_path.exists():
                    cache_path.write_bytes(buf)
                    logger.info(f"upload cached file={f.name} bytes={size_bytes} hash={file_hash} path={cache_path}")
                try:
                    grandpa_id = vs.add_file(cache_path, reset_collection=False, batch_size=int(batch_size))
                    item = {
                        "文件": f.name,
                        "状态": "成功",
                        "grandpa_id": grandpa_id,
                        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "hash": file_hash,
                    }
                    st.session_state["uploads_manifest"] = [
                        i for i in st.session_state["uploads_manifest"] if i.get("grandpa_id") != grandpa_id
                    ]
                    st.session_state["uploads_manifest"].insert(0, item)
                    _save_manifest(root, st.session_state["uploads_manifest"])
                    logger.info(
                        f"ingest ok file={f.name} bytes={size_bytes} hash={file_hash} cache_path={cache_path} grandpa_id={grandpa_id}"
                    )
                    ok_count += 1
                except Exception as e:
                    logger.error(
                        f"ingest fail file={f.name} bytes={size_bytes} hash={file_hash} cache_path={cache_path} error={e}"
                    )
                    st.error(f"{f.name} 入库失败：{e}")
                    fail_count += 1
            summary = f"入库完成：成功 {ok_count}，跳过 {skip_count}，失败 {fail_count}"
            status.update(label=summary, state="complete")
            if fail_count == 0:
                st.success(summary)
            else:
                st.warning(summary)
        logger.info(f"ingest done ok={ok_count} skip={skip_count} fail={fail_count}")

    uploads = st.session_state.get("uploads_manifest") or []
    with st.expander("同步已入库文档（用于历史入库但本地记录缺失）", expanded=False):
        max_unique = st.number_input("同步数量", min_value=1, max_value=200, value=50, step=1)
        scan_rows = st.number_input("扫描行数", min_value=300, max_value=50000, value=3000, step=300)
        do_sync = st.button("从 Milvus 同步", use_container_width=True)
        if do_sync:
            try:
                logger.info(f"sync start max_unique={int(max_unique)} scan_rows={int(scan_rows)}")
                gids = _scan_grandpa_ids_from_milvus(config_path, max_unique=int(max_unique), scan_rows=int(scan_rows))
                existing = {i.get("grandpa_id") for i in uploads if i.get("grandpa_id")}
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                added = 0
                for gid in gids:
                    if gid in existing:
                        continue
                    uploads.insert(
                        0,
                        {"文件": "(未知)", "状态": "已同步", "grandpa_id": gid, "时间": now},
                    )
                    existing.add(gid)
                    added += 1
                st.session_state["uploads_manifest"] = uploads
                _save_manifest(root, uploads)
                logger.info(f"sync done scanned={len(gids)} added={added} existing={len(existing)}")
                st.success("同步完成")
                st.rerun()
            except Exception as e:
                logger.error(f"sync fail error={e}")
                st.error(str(e))

    if uploads:
        st.subheader("已入库文件")
        for idx, row in enumerate(list(uploads)):
            grandpa_id = row.get("grandpa_id", "")
            if not grandpa_id:
                continue
            left, right = st.columns([4, 1])
            with left:
                file_name = row.get("文件", "")
                t = row.get("时间", "")
                if file_name in {"", "(未知)"}:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        new_name = st.text_input(
                            "文件名",
                            value=file_name if file_name else "",
                            placeholder="请输入文件名（用于展示）",
                            key=f"rename_{grandpa_id}_{idx}",
                            label_visibility="collapsed",
                        )
                        st.caption(f"{t}  |  grandpa_id={grandpa_id}")
                    with c2:
                        if st.button("保存名称", key=f"save_{grandpa_id}_{idx}", use_container_width=True):
                            if new_name.strip():
                                row["文件"] = new_name.strip()
                                _update_manifest_name(root, grandpa_id, new_name.strip())
                                st.session_state["uploads_manifest"] = _load_manifest(root)
                                logger.info(f"rename ok grandpa_id={grandpa_id} name={new_name.strip()}")
                                st.success("已保存")
                                st.rerun()
                            else:
                                st.warning("请输入名称")
                        if st.button("自动命名并保存", key=f"auto_{grandpa_id}_{idx}", use_container_width=True):
                            auto = _suggest_name_from_milvus(config_path, grandpa_id)
                            if auto:
                                row["文件"] = auto
                                _update_manifest_name(root, grandpa_id, auto)
                                st.session_state["uploads_manifest"] = _load_manifest(root)
                                logger.info(f"rename auto grandpa_id={grandpa_id} name={auto}")
                                st.success("已保存")
                                st.rerun()
                            else:
                                st.warning("无法自动生成名称")
                else:
                    st.write(f"{file_name}  |  {t}  |  grandpa_id={grandpa_id}")
            with right:
                if st.button("删除", key=f"del_{grandpa_id}_{idx}", use_container_width=True):
                    try:
                        vs = VectorStore()
                        vs.delete_by_grandpa_id(grandpa_id)
                        st.session_state["uploads_manifest"] = [i for i in uploads if i.get("grandpa_id") != grandpa_id]
                        _save_manifest(root, st.session_state["uploads_manifest"])
                        logger.info(f"delete ok grandpa_id={grandpa_id}")
                        st.success("删除完成")
                        st.rerun()
                    except Exception as e:
                        logger.error(f"delete fail grandpa_id={grandpa_id} error={e}")
                        st.error(str(e))

        st.subheader("记录文件")
        st.dataframe(uploads, use_container_width=True, hide_index=True)

