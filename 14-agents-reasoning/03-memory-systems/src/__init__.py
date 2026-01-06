"""
Memory Systems: Comprehensive Memory Framework for AI Agents

Core Idea:
    This module implements a complete memory system for AI agents, enabling
    them to maintain context, store long-term knowledge, and intelligently
    retrieve relevant information.

Mathematical Foundation:
    Memory retrieval combines multiple signals:
    
    $$score(m) = alpha * sim(q, m) + beta * recency(m) + gamma * importance(m)$$

References:
    - MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)
    - Generative Agents: Interactive Simulacra (Park et al., 2023)
"""

from __future__ import annotations

try:
    from .short_term_memory import (
        Message, MessageRole, ConversationBuffer, SlidingWindowMemory,
        SummaryMemory, TokenBasedMemory, ShortTermMemory,
        SimpleTokenCounter, SimpleSummarizer, create_conversation_memory,
    )
    from .long_term_memory import (
        MemoryEntry, MemoryType, VectorStore, InMemoryVectorStore,
        LongTermMemory, SimpleEmbedding,
    )
    from .memory_retrieval import (
        RetrievalResult, RetrievalStrategy, SimilarityRetrieval,
        RecencyRetrieval, ImportanceRetrieval, HybridRetrieval,
        MemoryRetriever, TimeDecay,
    )
except ImportError:
    from short_term_memory import (
        Message, MessageRole, ConversationBuffer, SlidingWindowMemory,
        SummaryMemory, TokenBasedMemory, ShortTermMemory,
        SimpleTokenCounter, SimpleSummarizer, create_conversation_memory,
    )
    from long_term_memory import (
        MemoryEntry, MemoryType, VectorStore, InMemoryVectorStore,
        LongTermMemory, SimpleEmbedding,
    )
    from memory_retrieval import (
        RetrievalResult, RetrievalStrategy, SimilarityRetrieval,
        RecencyRetrieval, ImportanceRetrieval, HybridRetrieval,
        MemoryRetriever, TimeDecay,
    )

__version__ = "1.0.0"
__author__ = "AI-Practices"

__all__ = [
    "__version__", "__author__",
    "Message", "MessageRole", "ConversationBuffer", "SlidingWindowMemory",
    "SummaryMemory", "TokenBasedMemory", "ShortTermMemory",
    "SimpleTokenCounter", "SimpleSummarizer", "create_conversation_memory",
    "MemoryEntry", "MemoryType", "VectorStore", "InMemoryVectorStore",
    "LongTermMemory", "SimpleEmbedding",
    "RetrievalResult", "RetrievalStrategy", "SimilarityRetrieval",
    "RecencyRetrieval", "ImportanceRetrieval", "HybridRetrieval",
    "MemoryRetriever", "TimeDecay",
]
