"""
Hierarchical topic-tree builder for a simulation session.

Embeds every (name + description) of every non-removed post via Gemini's
text-embedding model, runs average-link agglomerative clustering on those
vectors, and writes a TopicNode tree to the DB:

    root (level 0)
      └── topic clusters (level 1)   ← labeled by an LLM
            └── leaves = posts (level 2)

A post is a leaf TopicNode whose `member_post_ids` is exactly `[post_id]`.
An internal node's `member_post_ids` is the union of its descendants'.
"""

import math
import os
import random
import re
import asyncio
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from google import genai
from google.genai import types as genai_types

from models import DeliberationPost, TopicNode


EMBED_MODEL = "gemini-embedding-001"
LABEL_MODEL = "gemini-2.5-flash"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


# ── Math helpers (pure Python — sessions are small, no numpy needed) ─────────

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _cos_dist(a: List[float], b: List[float]) -> float:
    return 1.0 - _cosine(a, b)


def _mean_vec(vecs: List[List[float]]) -> List[float]:
    n = len(vecs)
    if n == 0:
        return []
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / n for i in range(dim)]


# ── Embedding ────────────────────────────────────────────────────────────────

async def _embed_text(text: str) -> List[float]:
    # Timeout + retry on transient 503/429 — one blip must not kill a long run.
    from simulation.roles.base import call_with_retry
    resp = await call_with_retry(lambda: _get_client().aio.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="CLUSTERING"),
    ), timeout=60)
    return list(resp.embeddings[0].values)


async def _embed_all(texts: List[str]) -> List[List[float]]:
    return await asyncio.gather(*[_embed_text(t) for t in texts])


# ── Clustering ───────────────────────────────────────────────────────────────

def _target_cluster_count(n: int) -> int:
    """Pick a reasonable top-level cluster count given n posts."""
    if n <= 3:
        return n
    return max(2, min(6, round(math.sqrt(n))))


def _kmeans_pp(vectors: List[List[float]], k: int, seed: int = 7, max_iters: int = 40) -> List[List[int]]:
    """
    K-means++ with cosine distance. Returns clusters as lists of original indices.
    Gives more balanced clusters than agglomerative single/average-link on dense
    embedding vectors (which tend to collapse into one mega-cluster).
    """
    n = len(vectors)
    if n == 0:
        return []
    if n <= k:
        return [[i] for i in range(n)]

    rng = random.Random(seed)

    # ── k-means++ seeding ────────────────────────────────────────────────
    first = rng.randrange(n)
    centroid_idxs = [first]
    while len(centroid_idxs) < k:
        dists_sq = []
        for i in range(n):
            min_d = min(_cos_dist(vectors[i], vectors[c]) for c in centroid_idxs)
            dists_sq.append(min_d ** 2)
        total = sum(dists_sq)
        if total == 0:
            # All points identical to existing centroids — pick any remaining
            remaining = [i for i in range(n) if i not in centroid_idxs]
            if not remaining:
                break
            centroid_idxs.append(rng.choice(remaining))
            continue
        threshold = rng.random() * total
        cumulative = 0.0
        for i, d in enumerate(dists_sq):
            cumulative += d
            if cumulative >= threshold and i not in centroid_idxs:
                centroid_idxs.append(i)
                break

    centroids = [list(vectors[i]) for i in centroid_idxs]

    # ── Lloyd iterations ─────────────────────────────────────────────────
    assignments = [0] * n
    for _ in range(max_iters):
        changed = False
        for i in range(n):
            best_c = 0
            best_d = float("inf")
            for c in range(len(centroids)):
                d = _cos_dist(vectors[i], centroids[c])
                if d < best_d:
                    best_d = d
                    best_c = c
            if assignments[i] != best_c:
                assignments[i] = best_c
                changed = True

        new_centroids = []
        for c in range(len(centroids)):
            members = [vectors[i] for i in range(n) if assignments[i] == c]
            if members:
                new_centroids.append(_mean_vec(members))
            else:
                # Empty cluster — re-seed to the point farthest from any centroid
                farthest = max(
                    range(n),
                    key=lambda i: min(_cos_dist(vectors[i], cc) for cc in centroids),
                )
                new_centroids.append(list(vectors[farthest]))
                assignments[farthest] = c
                changed = True
        centroids = new_centroids
        if not changed:
            break

    clusters: List[List[int]] = [[] for _ in range(len(centroids))]
    for i, c in enumerate(assignments):
        clusters[c].append(i)
    return [c for c in clusters if c]


# ── Labeling ─────────────────────────────────────────────────────────────────

async def _label_cluster(member_names: List[str]) -> str:
    """Ask Gemini for a 2-4 word topic label summarising the cluster."""
    sample = "\n- ".join(member_names[:12])
    prompt = (
        "You are labelling a cluster of deliberation posts with a SHORT topic name "
        "(2-4 words, title case, no punctuation, no quotes).\n\n"
        f"Posts in this cluster:\n- {sample}\n\n"
        "Return ONLY the topic label."
    )
    try:
        resp = await _get_client().aio.models.generate_content(
            model=LABEL_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=64,
                temperature=0.3,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        label = (resp.text or "").strip().strip('"').strip("'")
        label = re.sub(r"\s+", " ", label)
        if not label:
            return member_names[0][:40] if member_names else "Untitled Topic"
        return label[:60]
    except Exception:
        return member_names[0][:40] if member_names else "Untitled Topic"


# ── Public entry point ──────────────────────────────────────────────────────

async def rebuild_topic_tree(db: Session, session_id: str) -> dict:
    """
    Drop any existing topic tree for this session and rebuild it from current posts.
    Returns a summary dict: {"node_count": int, "leaf_count": int, "cluster_count": int}.
    """
    db.query(TopicNode).filter(TopicNode.session_id == session_id).delete()
    db.commit()

    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
        )
        .all()
    )

    if not posts:
        return {"node_count": 0, "leaf_count": 0, "cluster_count": 0}

    texts = [f"{p.name}. {p.description}" for p in posts]
    embeddings = await _embed_all(texts)

    k = _target_cluster_count(len(posts))
    clusters = _kmeans_pp(embeddings, k)

    root = TopicNode(
        session_id=session_id, parent_id=None,
        label="All Topics", level=0, is_leaf=False,
        member_post_ids=[p.id for p in posts],
        centroid=_mean_vec(embeddings),
    )
    db.add(root)
    db.flush()

    label_tasks = [
        _label_cluster([posts[i].name for i in member_idxs])
        for member_idxs in clusters
    ]
    cluster_labels = await asyncio.gather(*label_tasks)

    for member_idxs, label in zip(clusters, cluster_labels):
        cluster_members = [posts[i].id for i in member_idxs]
        cluster_centroid = _mean_vec([embeddings[i] for i in member_idxs])
        cluster_node = TopicNode(
            session_id=session_id, parent_id=root.id,
            label=label, level=1, is_leaf=False,
            member_post_ids=cluster_members, centroid=cluster_centroid,
        )
        db.add(cluster_node)
        db.flush()

        for i in member_idxs:
            leaf = TopicNode(
                session_id=session_id, parent_id=cluster_node.id,
                label=posts[i].name, level=2, is_leaf=True,
                member_post_ids=[posts[i].id], centroid=embeddings[i],
            )
            db.add(leaf)

    db.commit()

    counts = {
        "node_count": db.query(TopicNode).filter(TopicNode.session_id == session_id).count(),
        "leaf_count": len(posts),
        "cluster_count": len(clusters),
    }
    return counts
