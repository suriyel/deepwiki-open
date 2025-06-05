"""本地文档处理器 - 使用本地嵌入器"""

import os
import logging
from typing import Sequence
from copy import deepcopy
from tqdm import tqdm

from adalflow.core.types import Document
from adalflow.core.component import DataComponent

from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class LocalDocumentProcessor(DataComponent):
    """
    本地文档处理器，使用本地嵌入器为文档生成向量
    """

    def __init__(self, embedder) -> None:
        super().__init__()
        self.embedder = embedder

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        """处理文档，为每个文档生成嵌入向量"""
        output = deepcopy(documents)
        logger.info(f"Processing {len(output)} documents with local embedding")

        successful_docs = []
        expected_embedding_size = None

        for i, doc in enumerate(tqdm(output, desc="Generating local embeddings")):
            try:
                text = doc.text.strip()
                if not text:
                    logger.warning(f"Document {i} has empty text, skipping")
                    continue

                # 使用嵌入器生成向量
                result = self.embedder(input=f"passage: {text}")

                if result.data and len(result.data) > 0:
                    embedding = result.data[0].embedding

                    # 验证嵌入大小一致性
                    if expected_embedding_size is None:
                        expected_embedding_size = len(embedding)
                        logger.info(f"Expected embedding size: {expected_embedding_size}")
                    elif len(embedding) != expected_embedding_size:
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                        logger.warning(f"Document '{file_path}' has inconsistent embedding size {len(embedding)} != {expected_embedding_size}, skipping")
                        continue

                    # 关键：直接赋值给 document.vector
                    output[i].vector = embedding
                    successful_docs.append(output[i])
                else:
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Failed to get embedding for document '{file_path}', skipping")

            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.error(f"Error processing document '{file_path}': {e}, skipping")

        logger.info(f"Successfully processed {len(successful_docs)}/{len(output)} documents")
        return successful_docs