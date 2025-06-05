"""简化的本地嵌入器客户端"""

import os
import logging
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput

log = logging.getLogger(__name__)


class LocalEmbedderClient(ModelClient):
    """简化的本地嵌入器客户端，主要用于查询时的嵌入"""

    def __init__(self, model_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path or os.path.expanduser("D:\\01Code\\deepwiki-open\\models\\BAAI\\bge-small-en-v1___5")
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            if os.path.exists(self.model_path):
                self.model = SentenceTransformer(self.model_path)
                log.info(f"Loaded local model from: {self.model_path}")
            else:
                self.model = SentenceTransformer("intfloat/e5-small-v2")
                log.info("Loaded e5-small-v2 from HuggingFace")
        except Exception as e:
            log.error(f"Failed to load model: {str(e)}")
            raise

    def convert_inputs_to_api_kwargs(self, input: Any = None, model_kwargs: Dict = None,
                                     model_type: ModelType = None) -> Dict:
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            return {"input": input, **(model_kwargs or {})}
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """
        解析嵌入响应，必须实现的方法
        """
        try:
            # 使用 adalflow 的解析工具
            from adalflow.components.model_client.utils import parse_embedding_response
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def call(self, api_kwargs: Dict = None, model_type: ModelType = None):
        if model_type == ModelType.EMBEDDER:
            try:
                input_texts = api_kwargs.get("input", [])
                if not input_texts:
                    raise ValueError("No input provided")

                # 为查询文本添加前缀
                processed_texts = [f"query: {text}" for text in input_texts]

                embeddings = self.model.encode(
                    processed_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                # 创建模拟的 OpenAI 响应格式
                mock_response = type('MockResponse', (), {
                    'data': [
                        type('EmbeddingObject', (), {
                            'embedding': embedding.tolist(),
                            'index': i,
                            'object': 'embedding'
                        })()
                        for i, embedding in enumerate(embeddings)
                    ],
                    'model': 'local-e5-small-v2',
                    'object': 'list',
                    'usage': type('Usage', (), {
                        'prompt_tokens': sum(len(text.split()) for text in input_texts),
                        'total_tokens': sum(len(text.split()) for text in input_texts)
                    })()
                })()

                return mock_response

            except Exception as e:
                log.error(f"Error generating embeddings: {str(e)}")
                # 返回错误格式的响应
                return type('ErrorResponse', (), {
                    'data': [],
                    'error': str(e),
                    'model': 'local-e5-small-v2'
                })()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def acall(self, api_kwargs: Dict = None, model_type: ModelType = None):
        return self.call(api_kwargs, model_type)