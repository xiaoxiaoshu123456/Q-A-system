"""Vector store integration with Milvus and BGE-M3 embeddings.

- Encapsulates ingestion, deletion, retrieval, and LLM calling helpers.
- Uses Milvus standalone via pymilvus and hybrid search (dense + sparse).
- Embeddings are produced by FlagEmbedding BGEM3 model located under model/bge-m3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import Collection, utility

from init_database import (
    ChildChunk,
    MilvusService,
    _build_chunks,
    _call_llm_from_config,
    _connect_milvus,
    _embed_query,
    _embed_texts,
    _ensure_collection,
    _grandpa_id_for_doc,
    _hybrid_search,
    _lexical_weights_to_csr,
    _load_document,
    _parse_config,
    _unique_parents,
)


@dataclass(frozen=True)
class QueryResult:
    context: str
    parents: list[dict[str, Any]]
    hits: list[dict[str, Any]]


@dataclass(frozen=True)
class VectorStorePaths:
    root: Path
    config_path: Path
    model_path: Path

    @classmethod
    def resolve(cls, config_path: str | Path, model_path: str | Path) -> "VectorStorePaths":
        root = Path(__file__).resolve().parent
        cfg = (root / config_path).resolve()
        model = (root / model_path).resolve()
        return cls(root=root, config_path=cfg, model_path=model)


class CollectionManager:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def ensure(self, dense_dim: int, reset: bool) -> Collection:
        if utility.has_collection(self.collection_name) and not reset:
            return Collection(self.collection_name)
        return _ensure_collection(name=self.collection_name, dense_dim=dense_dim, reset=reset)

    def require(self) -> Collection:
        if not utility.has_collection(self.collection_name):
            raise RuntimeError(f"collection 不存在: {self.collection_name}")
        col = Collection(self.collection_name)
        col.load()
        return col


class IngestService:
    def __init__(self, paths: VectorStorePaths, retrieval_cfg: Any, collections: CollectionManager):
        self.paths = paths
        self.retrieval_cfg = retrieval_cfg
        self.collections = collections

    def add_file(self, file_path: str | Path, reset_collection: bool, batch_size: int) -> str:
        path = (self.paths.root / file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"找不到文档: {path}")

        doc_text = _load_document(path)
        grandpa_id = _grandpa_id_for_doc(path, doc_text)
        children: list[ChildChunk] = _build_chunks(doc_text, self.retrieval_cfg, grandpa_id)
        if not children:
            raise RuntimeError("未生成任何子块，无法写入")

        dense_vecs, lexical_weights, sparse_dim = _embed_texts(self.paths.model_path, [c.text for c in children], batch_size=batch_size)
        dense_dim = len(dense_vecs[0]) if dense_vecs else 0
        if dense_dim <= 0:
            raise RuntimeError("未生成稠密向量，无法写入")

        collection = self.collections.ensure(dense_dim=dense_dim, reset=reset_collection)

        ids = [c.id for c in children]
        grandpa_ids = [c.grandpa_id for c in children]
        texts = [c.text for c in children]
        parent_ids = [c.parent_id for c in children]
        parent_contexts = [c.parent_context for c in children]

        for start in range(0, len(children), batch_size):
            end = min(len(children), start + batch_size)
            b_ids = ids[start:end]
            b_grandpa_ids = grandpa_ids[start:end]
            b_texts = texts[start:end]
            b_dense = dense_vecs[start:end]
            b_sparse = _lexical_weights_to_csr(lexical_weights[start:end], sparse_dim)
            b_parent_ids = parent_ids[start:end]
            b_parent_contexts = parent_contexts[start:end]
            collection.insert([b_ids, b_grandpa_ids, b_texts, b_dense, b_sparse, b_parent_ids, b_parent_contexts])

        collection.flush()
        return grandpa_id


class DeleteService:
    def __init__(self, milvus: MilvusService):
        self.milvus = milvus

    def delete_by_grandpa_id(self, grandpa_id: str) -> None:
        self.milvus.delete_by_grandpa_id(grandpa_id)


class RetrievalService:
    def __init__(self, milvus_cfg: Any, collection_name: str, model_path: Path):
        self.milvus_cfg = milvus_cfg
        self.collection_name = collection_name
        self.model_path = model_path

    def query(self, question: str, top_k: int, dense_weight: float, sparse_weight: float) -> QueryResult:
        dense_q, sparse_q = _embed_query(self.model_path, question)
        hits = _hybrid_search(
            milvus_cfg=self.milvus_cfg,
            collection_name=self.collection_name,
            dense_vector=dense_q,
            sparse_vector=sparse_q,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        parents = _unique_parents(hits)
        context = "\n\n".join([p.get("parend_context", "") for p in parents if p.get("parend_context")])
        return QueryResult(context=context, parents=parents, hits=hits)


class LLMService:
    def __init__(self, config_path: Path):
        self.config_path = config_path

    def ask(self, question: str, context: str, phone: str) -> str:
        return _call_llm_from_config(self.config_path, question, context, phone)


class VectorStore:
    def __init__(self, config_path: str | Path = "config.ini", model_path: str | Path = Path("model") / "bge-m3"):
        self.paths = VectorStorePaths.resolve(config_path=config_path, model_path=model_path)

        if not self.paths.config_path.exists():
            raise FileNotFoundError(f"找不到配置文件: {self.paths.config_path}")
        if not self.paths.model_path.exists():
            raise FileNotFoundError(f"找不到模型路径: {self.paths.model_path}")

        self.milvus_cfg, self.retrieval_cfg = _parse_config(self.paths.config_path)
        _connect_milvus(self.milvus_cfg)
        self.milvus = MilvusService(self.milvus_cfg)

        self.collections = CollectionManager(self.milvus_cfg.collection_name)
        self.ingest = IngestService(self.paths, self.retrieval_cfg, self.collections)
        self.deleter = DeleteService(self.milvus)
        self.retriever = RetrievalService(self.milvus_cfg, self.milvus_cfg.collection_name, self.paths.model_path)
        self.llm = LLMService(self.paths.config_path)

    @property
    def collection_name(self) -> str:
        return self.milvus_cfg.collection_name

    def ensure_collection(self, dense_dim: int = 1024, reset: bool = False) -> Collection:
        return self.collections.ensure(dense_dim=dense_dim, reset=reset)

    def add_file(self, file_path: str | Path, reset_collection: bool = False, batch_size: int = 16) -> str:
        return self.ingest.add_file(file_path=file_path, reset_collection=reset_collection, batch_size=batch_size)

    def delete_by_grandpa_id(self, grandpa_id: str) -> None:
        self.deleter.delete_by_grandpa_id(grandpa_id)

    def query(
        self,
        question: str,
        top_k: int = 20,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.8,
    ) -> QueryResult:
        return self.retriever.query(question=question, top_k=top_k, dense_weight=dense_weight, sparse_weight=sparse_weight)

    def ask_llm(self, question: str, context: str, phone: str = "10086") -> str:
        return self.llm.ask(question=question, context=context, phone=phone)




