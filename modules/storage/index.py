"""generative_agents.storage.index"""

import os
import math
import time
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index import core as index_core
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from modules import utils


def _sanitize_embedding(embedding):
    """將向量中的 NaN/Inf 替換為 0，避免 JSON 序列化失敗"""
    return [0.0 if (v is None or not math.isfinite(v)) else v for v in embedding]


class _SafeEmbedding(BaseEmbedding):
    """包裝 embedding 模型，自動過濾 NaN/Inf，並在 Ollama 500 錯誤時回傳零向量"""

    # bge-m3 的標準維度；第一次成功後會自動更新
    _DEFAULT_DIM = 1024

    def __init__(self, embed_model):
        super().__init__(
            model_name=getattr(embed_model, "model_name", "safe_embedding"),
            embed_batch_size=getattr(embed_model, "embed_batch_size", 10),
        )
        object.__setattr__(self, "_inner_model", embed_model)
        object.__setattr__(self, "_dim", None)  # 快取向量維度

    def _fallback(self):
        dim = self._dim or self._DEFAULT_DIM
        return [0.0] * dim

    def _safe(self, vec):
        result = _sanitize_embedding(vec)
        if self._dim is None:
            object.__setattr__(self, "_dim", len(result))
        return result

    def _call(self, fn, *args, max_retry=3):
        """執行 fn，失敗時重試，最終回傳零向量而非拋出例外"""
        for attempt in range(max_retry):
            try:
                return self._safe(fn(*args))
            except Exception as e:
                print(f"[SafeEmbedding] 向量化失敗 (嘗試 {attempt+1}/{max_retry}): {e}")
                time.sleep(2)
        print("[SafeEmbedding] 已達重試上限，使用零向量替代")
        return self._fallback()

    def _get_query_embedding(self, query):
        return self._call(self._inner_model.get_query_embedding, query)

    def _get_text_embedding(self, text):
        return self._call(self._inner_model.get_text_embedding, text)

    def _get_text_embeddings(self, texts):
        results = []
        for text in texts:
            results.append(self._call(self._inner_model.get_text_embedding, text))
        return results

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text):
        return self._get_text_embedding(text)


class LlamaIndex:
    def __init__(self, embedding, path=None):
        self._config = {"max_nodes": 0}
        if embedding["type"] == "hugging_face":
            embed_model = HuggingFaceEmbedding(model_name=embedding["model"])
        elif embedding["type"] == "ollama":
            embed_model = OllamaEmbedding(
                model_name=embedding["model"],
                base_url=embedding["base_url"],
                ollama_additional_kwargs={"mirostat": 0},
            )
        else:
            raise NotImplementedError(
                "embedding type {} is not supported".format(embedding["type"])
            )

        Settings.embed_model = _SafeEmbedding(embed_model)
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        Settings.num_output = 1024
        Settings.context_window = 4096
        if path and os.path.exists(path):
            self._index = index_core.load_index_from_storage(
                index_core.StorageContext.from_defaults(persist_dir=path),
                show_progress=True,
            )
            self._config = utils.load_dict(os.path.join(path, "index_config.json"))
        else:
            self._index = index_core.VectorStoreIndex([], show_progress=True)
        self._path = path

    def add_node(
        self,
        text,
        metadata=None,
        exclude_llm_keys=None,
        exclude_embedding_keys=None,
        id=None,
    ):
        metadata = metadata or {}
        exclude_llm_keys = exclude_llm_keys or list(metadata.keys())
        exclude_embedding_keys = exclude_embedding_keys or list(metadata.keys())
        id = id or "node_" + str(self._config["max_nodes"])
        self._config["max_nodes"] += 1
        node = TextNode(
            text=text,
            id_=id,
            metadata=metadata,
            excluded_llm_metadata_keys=exclude_llm_keys,
            excluded_embed_metadata_keys=exclude_embedding_keys,
        )
        for attempt in range(3):
            try:
                self._index.insert_nodes([node])
                return node
            except Exception as e:
                print(f"LlamaIndex.add_node() caused an error: {e}")
                time.sleep(2)
        # 3 次失敗後仍回傳 node（不含向量），避免卡死
        print(f"LlamaIndex.add_node() skipped embedding for: {text[:50]}")
        return node

    def has_node(self, node_id):
        return node_id in self._index.docstore.docs

    def find_node(self, node_id):
        return self._index.docstore.docs[node_id]

    def get_nodes(self, filter=None):
        def _check(node):
            if not filter:
                return True
            return filter(node)

        return [n for n in self._index.docstore.docs.values() if _check(n)]

    def remove_nodes(self, node_ids, delete_from_docstore=True):
        self._index.delete_nodes(node_ids, delete_from_docstore=delete_from_docstore)

    def cleanup(self):
        now, remove_ids = utils.get_timer().get_date(), []
        for node_id, node in self._index.docstore.docs.items():
            create = utils.to_date(node.metadata["create"])
            expire = utils.to_date(node.metadata["expire"])
            if create > now or expire < now:
                remove_ids.append(node_id)
        self.remove_nodes(remove_ids)
        return remove_ids

    def retrieve(
        self,
        text,
        similarity_top_k=5,
        filters=None,
        node_ids=None,
        retriever_creator=None,
    ):
        while True:
            try:
                retriever_creator = retriever_creator or VectorIndexRetriever
                return retriever_creator(
                    self._index,
                    similarity_top_k=similarity_top_k,
                    filters=filters,
                    node_ids=node_ids,
                ).retrieve(text)
            except Exception as e:
                print(f"LlamaIndex.retrieve() caused an error: {e}")
                time.sleep(5)

    def query(
        self,
        text,
        similarity_top_k=5,
        text_qa_template=None,
        refine_template=None,
        filters=None,
        query_creator=None,
    ):
        kwargs = {
            "similarity_top_k": similarity_top_k,
            "text_qa_template": text_qa_template,
            "refine_template": refine_template,
            "filters": filters,
        }
        while True:
            try:
                if query_creator:
                    query_engine = query_creator(retriever=self._index.as_retriever(**kwargs))
                else:
                    query_engine = self._index.as_query_engine(**kwargs)
                return query_engine.query(text)
            except Exception as e:
                print(f"LlamaIndex.query() caused an error: {e}")
                time.sleep(5)

    def save(self, path=None):
        path = path or self._path
        self._index.storage_context.persist(path)
        utils.save_dict(self._config, os.path.join(path, "index_config.json"))

    @property
    def nodes_num(self):
        return len(self._index.docstore.docs)
