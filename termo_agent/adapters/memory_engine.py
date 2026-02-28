"""Semantic memory backed by ChromaDB with external embeddings."""

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb import Documents, Embeddings, EmbeddingFunction

AGENT_DIR = Path(__file__).parent
MEMORY_DIR = AGENT_DIR / "memory"
CHROMA_DIR = MEMORY_DIR / "chromadb"
LEGACY_FILE = MEMORY_DIR / "memory.md"

VALID_CATEGORIES = ("identity", "preference", "fact", "project", "user_profile")

# --- Embedding function (calls our LLM proxy) ---


class TermoEmbeddingFunction(EmbeddingFunction):
    """Calls our LLM proxy for embeddings (text-embedding-3-small via OpenRouter)."""

    def __init__(self, api_base: str, api_key: str, model: str = "openai/text-embedding-3-small"):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        payload = json.dumps({"model": self.model, "input": input}).encode()
        req = urllib.request.Request(
            f"{self.api_base}/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        sorted_embs = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in sorted_embs]


# --- Singleton ChromaDB client ---

_client = None
_collection = None


def _load_config():
    config_path = AGENT_DIR / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    MEMORY_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    config = _load_config()
    api_base = config.get("api_base", "https://api.termo.ai/v1")
    api_key = config.get("api_key", "")

    embed_fn = TermoEmbeddingFunction(api_base=api_base, api_key=api_key)

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(
        name="agent_memory",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Legacy migration
    _migrate_legacy()

    # One-time purge of old skill memories (now stored in skills table)
    _purge_skills_once()

    return _collection


def _purge_skills_once():
    """Remove category='skill' memories once, guarded by flag file."""
    flag = MEMORY_DIR / ".skills_migrated"
    if flag.exists():
        return
    try:
        results = _collection.get(where={"category": "skill"}, include=["documents"])
        ids = results.get("ids", [])
        if ids:
            _collection.delete(ids=ids)
    except Exception:
        pass
    flag.touch()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return f"mem_{uuid4().hex[:12]}"


# --- Legacy migration ---


def _migrate_legacy():
    """Migrate memory.md paragraphs into ChromaDB on first init."""
    if not LEGACY_FILE.exists():
        return
    content = LEGACY_FILE.read_text().strip()
    if not content:
        LEGACY_FILE.rename(LEGACY_FILE.with_suffix(".md.bak"))
        return

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if not paragraphs:
        LEGACY_FILE.rename(LEGACY_FILE.with_suffix(".md.bak"))
        return

    col = _collection
    now = _now_iso()

    for para in paragraphs:
        lower = para.lower()
        if any(kw in lower for kw in ("i am", "my name", "i'm called", "identity")):
            category = "identity"
        elif any(kw in lower for kw in ("prefer", "favorite", "like to", "don't like")):
            category = "preference"
        elif any(kw in lower for kw in ("project", "working on", "building", "repo")):
            category = "project"
        else:
            category = "fact"

        col.add(
            ids=[_new_id()],
            documents=[para],
            metadatas=[{
                "category": category,
                "created_at": now,
                "updated_at": now,
                "relationships": "[]",
            }],
        )

    LEGACY_FILE.rename(LEGACY_FILE.with_suffix(".md.bak"))


# --- Core functions ---


def remember(content: str, category: str = "fact", relationships: list | None = None) -> dict:
    """Add a memory. Dedup: if cosine distance < 0.15, update existing instead."""
    if category not in VALID_CATEGORIES:
        category = "fact"

    col = _get_collection()
    now = _now_iso()
    rels = relationships or []

    # Dedup check
    existing = col.query(query_texts=[content], n_results=1, include=["documents", "metadatas", "distances"])
    if existing["ids"] and existing["ids"][0] and existing["distances"][0][0] < 0.15:
        # Update existing memory
        mem_id = existing["ids"][0][0]
        old_meta = existing["metadatas"][0][0]
        old_rels = json.loads(old_meta.get("relationships", "[]"))
        merged_rels = old_rels + [r for r in rels if r not in old_rels]
        col.update(
            ids=[mem_id],
            documents=[content],
            metadatas=[{
                "category": category,
                "created_at": old_meta.get("created_at", now),
                "updated_at": now,
                "relationships": json.dumps(merged_rels),
            }],
        )
        # Update reverse relationships
        _update_reverse_rels(mem_id, rels)
        return {"status": "updated_existing", "id": mem_id, "content": content}

    # New memory
    mem_id = _new_id()
    col.add(
        ids=[mem_id],
        documents=[content],
        metadatas=[{
            "category": category,
            "created_at": now,
            "updated_at": now,
            "relationships": json.dumps(rels),
        }],
    )
    # Update reverse relationships
    _update_reverse_rels(mem_id, rels)
    return {"status": "created", "id": mem_id, "content": content}


def _update_reverse_rels(source_id: str, rels: list):
    """Add reverse relationships on target memories."""
    if not rels:
        return
    col = _get_collection()
    reverse_map = {"extends": "extended_by", "updates": "updated_by", "derives_from": "derived_to"}
    for rel in rels:
        target_id = rel.get("target_id", "")
        rel_type = rel.get("type", "extends")
        if not target_id:
            continue
        try:
            target = col.get(ids=[target_id], include=["metadatas"])
            if target["ids"]:
                meta = target["metadatas"][0]
                existing_rels = json.loads(meta.get("relationships", "[]"))
                reverse_rel = {"type": reverse_map.get(rel_type, rel_type), "target_id": source_id}
                if reverse_rel not in existing_rels:
                    existing_rels.append(reverse_rel)
                    col.update(
                        ids=[target_id],
                        metadatas=[{**meta, "relationships": json.dumps(existing_rels), "updated_at": _now_iso()}],
                    )
        except Exception:
            pass


def recall(query: str, limit: int = 5, category: str | None = None) -> list[dict]:
    """Semantic search over memories."""
    col = _get_collection()
    where = {"category": category} if category else None
    results = col.query(
        query_texts=[query],
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    memories = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        memories.append({
            "id": doc_id,
            "content": results["documents"][0][i],
            "category": meta.get("category", "fact"),
            "similarity": round(1 - dist, 4),
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "relationships": json.loads(meta.get("relationships", "[]")),
        })
    return memories


def forget(query: str) -> dict:
    """Delete the closest matching memory."""
    col = _get_collection()
    results = col.query(query_texts=[query], n_results=1, include=["documents", "distances"])
    if not results["ids"] or not results["ids"][0]:
        return {"deleted": False, "error": "no matching memory found"}
    mem_id = results["ids"][0][0]
    content = results["documents"][0][0]
    col.delete(ids=[mem_id])
    return {"deleted": True, "id": mem_id, "content": content}


def update_memory_entry(query: str, new_content: str) -> dict:
    """Update the closest matching memory's content."""
    col = _get_collection()
    results = col.query(query_texts=[query], n_results=1, include=["documents", "metadatas", "distances"])
    if not results["ids"] or not results["ids"][0]:
        return {"updated": False, "error": "no matching memory found"}
    mem_id = results["ids"][0][0]
    meta = results["metadatas"][0][0]
    meta["updated_at"] = _now_iso()
    col.update(ids=[mem_id], documents=[new_content], metadatas=[meta])
    return {"updated": True, "id": mem_id, "old_content": results["documents"][0][0], "new_content": new_content}


def get_all_memories() -> list[dict]:
    """Return all memories as structured dicts (for web UI + graph)."""
    col = _get_collection()
    results = col.get(include=["documents", "metadatas"])
    memories = []
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        memories.append({
            "id": doc_id,
            "content": results["documents"][i],
            "category": meta.get("category", "fact"),
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "relationships": json.loads(meta.get("relationships", "[]")),
        })
    return memories


def get_identity_and_preference_memories() -> list[dict]:
    """Return identity + preference memories for system prompt injection."""
    col = _get_collection()
    memories = []
    for cat in ("identity", "preference"):
        try:
            results = col.get(where={"category": cat}, include=["documents", "metadatas"])
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                memories.append({
                    "id": doc_id,
                    "content": results["documents"][i],
                    "category": cat,
                    "created_at": meta.get("created_at", ""),
                    "updated_at": meta.get("updated_at", ""),
                })
        except Exception:
            pass
    return memories


# --- Backward-compat wrappers ---


def load_memory() -> str:
    """Legacy compat: return all memories as text."""
    try:
        memories = get_all_memories()
        if memories:
            return "\n\n".join(f"[{m['category']}] {m['content']}" for m in memories)
    except Exception:
        pass
    # Fallback to legacy file
    MEMORY_DIR.mkdir(exist_ok=True)
    if LEGACY_FILE.exists():
        return LEGACY_FILE.read_text()
    return ""


def save_memory_legacy(content: str) -> None:
    """Legacy compat: write raw text to memory.md."""
    MEMORY_DIR.mkdir(exist_ok=True)
    LEGACY_FILE.write_text(content)


def purge_skill_memories() -> int:
    """Remove all memories with category='skill' from ChromaDB.

    Called once during initialization, guarded by a flag file.
    Returns the number of purged entries.
    """
    flag = MEMORY_DIR / ".skills_migrated"
    if flag.exists():
        return 0

    try:
        col = _get_collection()
        results = col.get(where={"category": "skill"}, include=["documents"])
        ids = results.get("ids", [])
        if ids:
            col.delete(ids=ids)
        flag.touch()
        return len(ids)
    except Exception:
        # Don't block startup if this fails
        return 0
