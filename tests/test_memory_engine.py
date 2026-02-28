"""Tests for termo_agent.adapters.memory_engine — ChromaDB-backed semantic memory."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock chromadb before importing memory_engine (not installed in dev)
# ---------------------------------------------------------------------------

_fake_chromadb = types.ModuleType("chromadb")
_fake_chromadb.Documents = list
_fake_chromadb.Embeddings = list
_fake_chromadb.EmbeddingFunction = object  # base class for TermoEmbeddingFunction
_fake_chromadb.PersistentClient = MagicMock  # never actually called — we mock _get_collection


class FakeCollection:
    """In-memory stand-in for chromadb.Collection."""

    def __init__(self):
        self._docs: dict[str, dict] = {}  # id -> {document, metadata}

    def add(self, ids, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._docs[doc_id] = {"document": documents[i], "metadata": metadatas[i]}

    def get(self, ids=None, where=None, include=None):
        if ids:
            matched = [(k, v) for k, v in self._docs.items() if k in ids]
        elif where:
            matched = [
                (k, v) for k, v in self._docs.items()
                if all(v["metadata"].get(wk) == wv for wk, wv in where.items())
            ]
        else:
            matched = list(self._docs.items())
        return {
            "ids": [m[0] for m in matched],
            "documents": [m[1]["document"] for m in matched],
            "metadatas": [m[1]["metadata"] for m in matched],
        }

    def query(self, query_texts, n_results=5, where=None, include=None):
        items = list(self._docs.items())
        if where:
            items = [
                (k, v) for k, v in items
                if all(v["metadata"].get(wk) == wv for wk, wv in where.items())
            ]
        items = items[:n_results]
        return {
            "ids": [[k for k, _ in items]],
            "documents": [[v["document"] for _, v in items]],
            "metadatas": [[v["metadata"] for _, v in items]],
            "distances": [[0.5] * len(items)],
        }

    def update(self, ids, documents=None, metadatas=None):
        for i, doc_id in enumerate(ids):
            if doc_id in self._docs:
                if documents:
                    self._docs[doc_id]["document"] = documents[i]
                if metadatas:
                    self._docs[doc_id]["metadata"] = metadatas[i]

    def delete(self, ids):
        for doc_id in ids:
            self._docs.pop(doc_id, None)


# Install fake chromadb before any import of memory_engine
sys.modules.setdefault("chromadb", _fake_chromadb)

# Now we can import
from termo_agent.adapters import memory_engine as me


@pytest.fixture(autouse=True)
def _reset_module(tmp_path):
    """Reset memory_engine singletons and point AGENT_DIR to a temp dir."""
    me._client = None
    me._collection = None
    # Point module paths to tmp so mkdir/file ops don't hit /home/sprite
    me.AGENT_DIR = tmp_path
    me.MEMORY_DIR = tmp_path / "memory"
    me.CHROMA_DIR = tmp_path / "memory" / "chromadb"
    me.LEGACY_FILE = tmp_path / "memory" / "memory.md"
    yield
    me._client = None
    me._collection = None


@pytest.fixture
def fake_col():
    """Provide a FakeCollection and patch _get_collection to return it."""
    col = FakeCollection()
    with patch.object(me, "_get_collection", return_value=col):
        yield col


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------

class TestRemember:
    def test_creates_new_memory(self, fake_col):
        result = me.remember("User likes dark mode", "preference")
        assert result["status"] == "created"
        assert result["content"] == "User likes dark mode"
        assert len(fake_col._docs) == 1
        doc = list(fake_col._docs.values())[0]
        assert doc["metadata"]["category"] == "preference"

    def test_dedup_updates_existing(self, fake_col):
        fake_col.add(
            ids=["existing_1"],
            documents=["User likes dark mode"],
            metadatas=[{"category": "preference", "created_at": "t0", "updated_at": "t0", "relationships": "[]"}],
        )
        original_query = fake_col.query

        def fake_query(query_texts, n_results=1, include=None):
            result = original_query(query_texts, n_results, include=include)
            result["distances"] = [[0.05]]  # Below 0.15 threshold
            return result
        fake_col.query = fake_query

        result = me.remember("User likes dark mode a lot", "preference")
        assert result["status"] == "updated_existing"
        assert result["id"] == "existing_1"

    def test_invalid_category_defaults_to_fact(self, fake_col):
        result = me.remember("Some info", "invalid_cat")
        assert result["status"] == "created"
        doc = list(fake_col._docs.values())[0]
        assert doc["metadata"]["category"] == "fact"


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

class TestRecall:
    def test_returns_matching_memories(self, fake_col):
        me.remember("Likes Python", "preference")
        me.remember("Working on Termo project", "project")
        results = me.recall("Python")
        assert len(results) == 2
        assert all("similarity" in m for m in results)
        assert all("id" in m for m in results)

    def test_category_filter(self, fake_col):
        me.remember("Likes Python", "preference")
        me.remember("Building an API", "project")
        results = me.recall("test", category="project")
        assert len(results) == 1
        assert results[0]["category"] == "project"

    def test_empty_db_returns_empty(self, fake_col):
        results = me.recall("anything")
        assert results == []


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------

class TestForget:
    def test_deletes_closest_match(self, fake_col):
        me.remember("Delete me please", "fact")
        assert len(fake_col._docs) == 1
        result = me.forget("Delete me")
        assert result["deleted"] is True
        assert len(fake_col._docs) == 0

    def test_no_match_returns_error(self, fake_col):
        result = me.forget("nothing here")
        assert result["deleted"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# update_memory_entry
# ---------------------------------------------------------------------------

class TestUpdateMemoryEntry:
    def test_updates_content(self, fake_col):
        me.remember("Old content", "fact")
        result = me.update_memory_entry("Old content", "New content")
        assert result["updated"] is True
        assert result["old_content"] == "Old content"
        assert result["new_content"] == "New content"

    def test_no_match_returns_error(self, fake_col):
        result = me.update_memory_entry("nothing", "new")
        assert result["updated"] is False


# ---------------------------------------------------------------------------
# get_all_memories
# ---------------------------------------------------------------------------

class TestGetAllMemories:
    def test_returns_all(self, fake_col):
        me.remember("Fact one", "fact")
        me.remember("Fact two", "project")
        all_mems = me.get_all_memories()
        assert len(all_mems) == 2
        categories = {m["category"] for m in all_mems}
        assert categories == {"fact", "project"}

    def test_empty_db(self, fake_col):
        assert me.get_all_memories() == []


# ---------------------------------------------------------------------------
# get_identity_and_preference_memories
# ---------------------------------------------------------------------------

class TestIdentityAndPreference:
    def test_filters_categories(self, fake_col):
        me.remember("I am a coding bot", "identity")
        me.remember("User prefers dark mode", "preference")
        me.remember("Random fact", "fact")
        mems = me.get_identity_and_preference_memories()
        categories = {m["category"] for m in mems}
        assert categories == {"identity", "preference"}
        assert len(mems) == 2


# ---------------------------------------------------------------------------
# load_memory (legacy compat)
# ---------------------------------------------------------------------------

class TestLoadMemory:
    def test_returns_text_from_chromadb(self, fake_col):
        me.remember("Hello world", "fact")
        text = me.load_memory()
        assert "Hello world" in text
        assert "[fact]" in text

    def test_empty_returns_empty(self, fake_col):
        text = me.load_memory()
        assert text == ""


# ---------------------------------------------------------------------------
# VALID_CATEGORIES constant
# ---------------------------------------------------------------------------

class TestValidCategories:
    def test_expected_categories(self):
        assert "identity" in me.VALID_CATEGORIES
        assert "preference" in me.VALID_CATEGORIES
        assert "fact" in me.VALID_CATEGORIES
        assert "project" in me.VALID_CATEGORIES
        assert "user_profile" in me.VALID_CATEGORIES
