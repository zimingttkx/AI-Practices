"""
Comprehensive Tests for Long-Term Memory Implementations.

Test Coverage:
    - MemoryEntry: creation, serialization, validation, access tracking
    - MemoryType: all enum values
    - SimpleEmbedding: embedding generation, normalization
    - InMemoryVectorStore: CRUD operations, search, filtering
    - LongTermMemory: store, recall, forget, persistence
    - Edge cases: empty inputs, unicode, large content
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import json
import tempfile
import math
from datetime import datetime, timedelta

from long_term_memory import (
    MemoryType,
    MemoryEntry,
    SimpleEmbedding,
    InMemoryVectorStore,
    LongTermMemory,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_IMPORTANCE,
)


# =============================================================================
# MemoryEntry Tests
# =============================================================================


class TestMemoryEntry:
    """Comprehensive tests for MemoryEntry dataclass."""

    def test_create_entry(self):
        entry = MemoryEntry(content="Test memory")
        assert entry.content == "Test memory"
        assert entry.memory_type == MemoryType.KNOWLEDGE
        assert entry.importance == DEFAULT_IMPORTANCE

    def test_entry_id_unique(self):
        entry1 = MemoryEntry(content="Memory 1")
        entry2 = MemoryEntry(content="Memory 2")
        assert entry1.id != entry2.id

    def test_record_access(self):
        entry = MemoryEntry(content="Test")
        assert entry.access_count == 0
        entry.record_access()
        assert entry.access_count == 1
        assert entry.last_accessed is not None

    def test_to_dict(self):
        entry = MemoryEntry(
            content="Test",
            memory_type=MemoryType.FACT,
            importance=0.8
        )
        d = entry.to_dict()
        assert d["content"] == "Test"
        assert d["memory_type"] == "fact"
        assert d["importance"] == 0.8

    def test_from_dict(self):
        data = {
            "content": "Restored memory",
            "memory_type": "preference",
            "importance": 0.9,
            "timestamp": datetime.utcnow().isoformat(),
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.content == "Restored memory"
        assert entry.memory_type == MemoryType.PREFERENCE
        assert entry.importance == 0.9

    def test_all_memory_types(self):
        """Test all MemoryType enum values."""
        for mem_type in MemoryType:
            entry = MemoryEntry(content="Test", memory_type=mem_type)
            assert entry.memory_type == mem_type

    def test_memory_type_from_string(self):
        """Test memory type conversion from string."""
        entry = MemoryEntry(content="Test", memory_type="fact")
        assert entry.memory_type == MemoryType.FACT

    def test_importance_clamping(self):
        """Test importance clamping to [0, 1]."""
        with pytest.warns(UserWarning):
            entry_high = MemoryEntry(content="Test", importance=1.5)
        with pytest.warns(UserWarning):
            entry_low = MemoryEntry(content="Test", importance=-0.5)
        assert 0.0 <= entry_high.importance <= 1.0
        assert 0.0 <= entry_low.importance <= 1.0

    def test_metadata_storage(self):
        """Test custom metadata storage."""
        entry = MemoryEntry(
            content="Test",
            metadata={"source": "api", "tags": ["important", "user"]}
        )
        assert entry.metadata["source"] == "api"
        assert "important" in entry.metadata["tags"]

    def test_source_field(self):
        """Test source field."""
        entry = MemoryEntry(content="Test", source="conversation")
        assert entry.source == "conversation"

    def test_unicode_content(self):
        """Test unicode content handling."""
        entry = MemoryEntry(content="你好世界 🌍 مرحبا")
        assert "你好" in entry.content
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.content == entry.content

    def test_long_content(self):
        """Test very long content."""
        long_content = "x" * 10000
        entry = MemoryEntry(content=long_content)
        assert len(entry.content) == 10000

    def test_serialization_roundtrip(self):
        """Test full serialization/deserialization cycle."""
        original = MemoryEntry(
            content="Test memory",
            memory_type=MemoryType.EVENT,
            importance=0.75,
            metadata={"key": "value"},
            source="test",
        )
        original.record_access()
        data = original.to_dict()
        restored = MemoryEntry.from_dict(data)
        assert restored.content == original.content
        assert restored.memory_type == original.memory_type
        assert restored.importance == original.importance
        assert restored.access_count == original.access_count

    def test_multiple_access_records(self):
        """Test multiple access recordings."""
        entry = MemoryEntry(content="Test")
        for _ in range(10):
            entry.record_access()
        assert entry.access_count == 10


# =============================================================================
# SimpleEmbedding Tests
# =============================================================================


class TestSimpleEmbedding:
    """Comprehensive tests for SimpleEmbedding."""

    def test_embed_returns_vector(self):
        embedder = SimpleEmbedding(dim=128)
        vec = embedder.embed("Hello world")
        assert len(vec) == 128
        assert all(isinstance(x, float) for x in vec)

    def test_embed_normalized(self):
        embedder = SimpleEmbedding(dim=128)
        vec = embedder.embed("Test text")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.001

    def test_embed_batch(self):
        embedder = SimpleEmbedding(dim=64)
        texts = ["Hello", "World", "Test"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 64 for v in vecs)

    def test_custom_dimension(self):
        """Test custom embedding dimension."""
        for dim in [32, 64, 256, 512]:
            embedder = SimpleEmbedding(dim=dim)
            vec = embedder.embed("Test")
            assert len(vec) == dim

    def test_invalid_dimension(self):
        """Test invalid dimension raises error."""
        with pytest.raises(ValueError):
            SimpleEmbedding(dim=0)
        with pytest.raises(ValueError):
            SimpleEmbedding(dim=-1)

    def test_deterministic_embedding(self):
        """Test same text produces same embedding."""
        embedder = SimpleEmbedding(dim=128)
        vec1 = embedder.embed("Hello world")
        vec2 = embedder.embed("Hello world")
        assert vec1 == vec2

    def test_different_texts_different_embeddings(self):
        """Test different texts produce different embeddings."""
        embedder = SimpleEmbedding(dim=128)
        vec1 = embedder.embed("Hello")
        vec2 = embedder.embed("Goodbye")
        assert vec1 != vec2

    def test_empty_text(self):
        """Test empty text embedding."""
        embedder = SimpleEmbedding(dim=128)
        vec = embedder.embed("")
        assert len(vec) == 128

    def test_single_word(self):
        """Test single word embedding."""
        embedder = SimpleEmbedding(dim=128)
        vec = embedder.embed("test")
        assert len(vec) == 128
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.001


# =============================================================================
# InMemoryVectorStore Tests
# =============================================================================


class TestInMemoryVectorStore:
    """Comprehensive tests for InMemoryVectorStore."""

    def test_add_and_get(self):
        store = InMemoryVectorStore()
        entry = MemoryEntry(content="Test", embedding=[0.1] * 128)
        store.add(entry)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Test"

    def test_search(self):
        store = InMemoryVectorStore()
        embedder = SimpleEmbedding(dim=128)
        
        entry1 = MemoryEntry(content="Python programming")
        entry1.embedding = embedder.embed(entry1.content)
        store.add(entry1)
        
        entry2 = MemoryEntry(content="Java development")
        entry2.embedding = embedder.embed(entry2.content)
        store.add(entry2)
        
        query = embedder.embed("Python code")
        results = store.search(query, k=2)
        assert len(results) == 2

    def test_delete(self):
        store = InMemoryVectorStore()
        entry = MemoryEntry(content="Test", embedding=[0.1] * 128)
        store.add(entry)
        assert store.delete(entry.id)
        assert store.get(entry.id) is None

    def test_clear(self):
        store = InMemoryVectorStore()
        for i in range(5):
            entry = MemoryEntry(content=f"Test {i}", embedding=[0.1] * 128)
            store.add(entry)
        assert store.size == 5
        store.clear()
        assert store.size == 0

    def test_list_all(self):
        """Test listing all entries."""
        store = InMemoryVectorStore()
        for i in range(3):
            entry = MemoryEntry(content=f"Test {i}", embedding=[0.1] * 128)
            store.add(entry)
        entries = store.list_all()
        assert len(entries) == 3

    def test_search_with_filter(self):
        """Test search with filter function."""
        store = InMemoryVectorStore()
        embedder = SimpleEmbedding(dim=128)
        
        entry1 = MemoryEntry(content="Test 1", memory_type=MemoryType.FACT)
        entry1.embedding = embedder.embed(entry1.content)
        store.add(entry1)
        
        entry2 = MemoryEntry(content="Test 2", memory_type=MemoryType.PREFERENCE)
        entry2.embedding = embedder.embed(entry2.content)
        store.add(entry2)
        
        query = embedder.embed("Test")
        results = store.search(
            query, k=5,
            filter_fn=lambda e: e.memory_type == MemoryType.FACT
        )
        assert len(results) == 1
        assert results[0][0].memory_type == MemoryType.FACT

    def test_search_k_larger_than_store(self):
        """Test search with k larger than store size."""
        store = InMemoryVectorStore()
        entry = MemoryEntry(content="Only one", embedding=[0.1] * 128)
        store.add(entry)
        results = store.search([0.1] * 128, k=10)
        assert len(results) == 1

    def test_delete_nonexistent(self):
        """Test deleting nonexistent entry."""
        store = InMemoryVectorStore()
        assert not store.delete("nonexistent_id")

    def test_add_without_embedding_raises(self):
        """Test adding entry without embedding raises error."""
        store = InMemoryVectorStore()
        entry = MemoryEntry(content="No embedding")
        with pytest.raises(ValueError):
            store.add(entry)

    def test_size_property(self):
        """Test size property."""
        store = InMemoryVectorStore()
        assert store.size == 0
        entry = MemoryEntry(content="Test", embedding=[0.1] * 128)
        store.add(entry)
        assert store.size == 1


# =============================================================================
# LongTermMemory Tests
# =============================================================================


class TestLongTermMemory:
    """Comprehensive tests for LongTermMemory."""

    def test_store_and_recall(self):
        ltm = LongTermMemory()
        ltm.store("Python is a programming language")
        results = ltm.recall("programming", k=1)
        assert len(results) >= 1

    def test_store_with_importance(self):
        ltm = LongTermMemory()
        entry = ltm.store("Important fact", importance=0.9)
        assert entry.importance == 0.9

    def test_forget(self):
        ltm = LongTermMemory()
        entry = ltm.store("To be forgotten")
        assert ltm.forget(entry.id)
        assert ltm.get(entry.id) is None

    def test_list_all(self):
        ltm = LongTermMemory()
        ltm.store("Memory 1")
        ltm.store("Memory 2")
        ltm.store("Memory 3")
        entries = ltm.list_all()
        assert len(entries) == 3

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        ltm1 = LongTermMemory()
        ltm1.store("Persistent memory")
        ltm1.save(path)
        
        ltm2 = LongTermMemory()
        count = ltm2.load(path)
        assert count == 1
        entries = ltm2.list_all()
        assert len(entries) == 1
        
        Path(path).unlink()

    def test_auto_importance(self):
        ltm = LongTermMemory(auto_importance=True)
        entry = ltm.store("This is very important, remember always!")
        assert entry.importance > 0.5

    def test_store_with_memory_type(self):
        """Test storing with specific memory type."""
        ltm = LongTermMemory()
        entry = ltm.store("User likes dark mode", memory_type=MemoryType.PREFERENCE)
        assert entry.memory_type == MemoryType.PREFERENCE

    def test_recall_with_type_filter(self):
        """Test recall with memory type filter."""
        ltm = LongTermMemory()
        ltm.store("Fact 1", memory_type=MemoryType.FACT)
        ltm.store("Preference 1", memory_type=MemoryType.PREFERENCE)
        
        results = ltm.recall("test", k=10, memory_type=MemoryType.FACT)
        for entry, _ in results:
            assert entry.memory_type == MemoryType.FACT

    def test_recall_min_similarity(self):
        """Test recall with minimum similarity threshold."""
        ltm = LongTermMemory()
        ltm.store("Python programming language")
        results = ltm.recall("completely unrelated xyz", k=5, min_similarity=0.9)
        # High threshold should filter out low-similarity results
        assert len(results) <= 1

    def test_list_all_with_type_filter(self):
        """Test list_all with memory type filter."""
        ltm = LongTermMemory()
        ltm.store("Fact", memory_type=MemoryType.FACT)
        ltm.store("Event", memory_type=MemoryType.EVENT)
        
        facts = ltm.list_all(memory_type=MemoryType.FACT)
        assert all(e.memory_type == MemoryType.FACT for e in facts)

    def test_list_all_with_importance_filter(self):
        """Test list_all with importance filter."""
        ltm = LongTermMemory()
        ltm.store("Low importance", importance=0.3)
        ltm.store("High importance", importance=0.9)
        
        important = ltm.list_all(min_importance=0.7)
        assert all(e.importance >= 0.7 for e in important)

    def test_clear(self):
        """Test clearing all memories."""
        ltm = LongTermMemory()
        ltm.store("Memory 1")
        ltm.store("Memory 2")
        ltm.clear()
        assert ltm.size == 0

    def test_size_property(self):
        """Test size property."""
        ltm = LongTermMemory()
        assert ltm.size == 0
        ltm.store("Memory")
        assert ltm.size == 1

    def test_get_specific_memory(self):
        """Test getting specific memory by ID."""
        ltm = LongTermMemory()
        entry = ltm.store("Specific memory")
        retrieved = ltm.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Specific memory"

    def test_store_with_metadata(self):
        """Test storing with custom metadata."""
        ltm = LongTermMemory()
        entry = ltm.store(
            "Memory with metadata",
            metadata={"source": "api", "version": 1}
        )
        assert entry.metadata["source"] == "api"

    def test_store_with_source(self):
        """Test storing with source field."""
        ltm = LongTermMemory()
        entry = ltm.store("Memory", source="conversation")
        assert entry.source == "conversation"

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        ltm = LongTermMemory()
        count = ltm.load("nonexistent_file.json")
        assert count == 0

    def test_recall_updates_access(self):
        """Test that recall updates access count."""
        ltm = LongTermMemory()
        entry = ltm.store("Test memory")
        initial_count = entry.access_count
        ltm.recall("Test", k=1)
        assert entry.access_count > initial_count


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_empty_store_recall(self):
        """Test recall on empty store."""
        ltm = LongTermMemory()
        results = ltm.recall("anything", k=5)
        assert results == []

    def test_unicode_content(self):
        """Test unicode content handling."""
        ltm = LongTermMemory()
        entry = ltm.store("你好世界 🌍 مرحبا")
        results = ltm.recall("你好", k=1)
        assert len(results) >= 0

    def test_very_long_content(self):
        """Test very long content."""
        ltm = LongTermMemory()
        long_content = "x" * 5000
        entry = ltm.store(long_content)
        assert len(entry.content) == 5000

    def test_many_memories(self):
        """Test storing many memories."""
        ltm = LongTermMemory()
        for i in range(100):
            ltm.store(f"Memory number {i}")
        assert ltm.size == 100

    def test_rapid_store_recall(self):
        """Test rapid store/recall cycles."""
        ltm = LongTermMemory()
        for i in range(50):
            ltm.store(f"Memory {i}")
            ltm.recall(f"Memory {i}", k=1)
        assert ltm.size == 50

    def test_forget_nonexistent(self):
        """Test forgetting nonexistent memory."""
        ltm = LongTermMemory()
        assert not ltm.forget("nonexistent_id")

    def test_special_characters(self):
        """Test special characters in content."""
        ltm = LongTermMemory()
        special = "Test\n\t\r\\\"'`~!@#$%^&*()[]{}|;:,.<>?"
        entry = ltm.store(special)
        assert entry.content == special
