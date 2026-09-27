"""
Decision-tree user model (replaces the persona table as the predictor).

Each agent's model is a decision tree learned from its clickstream — every
recommendation logged as (post embedding -> engaged/skipped). The tree both models
the user (its leaves are data-discovered "topics", labelled post-hoc from the real
posts that fall in them) and scores candidates for the recommender.

Pipeline: PCA-reduce the post embeddings (speed) -> fit a per-agent
DecisionTreeClassifier (entropy criterion, min_samples_leaf to avoid over-chopping)
-> predict engage probability. All clickstream events are persisted, so a retrain
re-derives the whole hierarchy from scratch (automatic event redistribution).
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy.orm import Session
from google import genai
from google.genai import types as genai_types

from models import (
    PostEmbedding, ClickstreamEvent, DeliberationPost, UserModelSnapshot,
)
from simulation.ground_truth import load_ground_truth, gt_engage


PCA_COMPONENTS = 50
# Lowered 8 -> 4: the tree is the primary ranking signal, so the sooner it exists the
# sooner the skip rate can start falling. 4 events is thin but still beats cosine fit.
MIN_EVENTS_TO_TRAIN = 4          # fast profile (default)
MIN_EVENTS_TO_TRAIN_SLOW = 8     # pre-2026-09 baseline, for ablation


def set_learning_profile(fast: bool) -> None:
    """Toggle the learning-speed knobs so the effect can be ablated cleanly."""
    global MIN_EVENTS_TO_TRAIN
    MIN_EVENTS_TO_TRAIN = 4 if fast else MIN_EVENTS_TO_TRAIN_SLOW
MIN_SAMPLES_LEAF = 3
LABEL_MODEL = "gemini-2.5-flash"

# Live trained models per (session_id, agent_id): {"pca", "clf", "const"}.
_MODELS: dict = {}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


# ── Recording ─────────────────────────────────────────────────────────────────

def record_event(db: Session, session_id: str, agent_id: str, post_id: str,
                 vec: Optional[list], label: int, tick: int) -> None:
    """Persist one clickstream event (and the post's embedding, deduped by post_id)."""
    if vec is not None:
        db.merge(PostEmbedding(post_id=post_id, session_id=session_id, vector=list(vec)))
    db.add(ClickstreamEvent(
        session_id=session_id, agent_id=agent_id, post_id=post_id,
        label=int(label), sim_tick=tick,
    ))
    db.commit()


# ── Prediction (used by the recommender) ──────────────────────────────────────

def predict_engage_prob(session_id: str, agent_id: str, vec: list) -> Optional[float]:
    """Predicted probability the agent engages a post with embedding `vec`, or None
    if there is no trained model yet (recommender then falls back)."""
    m = _MODELS.get((session_id, agent_id))
    if not m:
        return None
    if m.get("const") is not None:
        return float(m["const"])
    x = m["pca"].transform([vec])[0] if m["pca"] is not None else np.asarray(vec)
    clf = m["clf"]
    proba = clf.predict_proba([x])[0]
    classes = list(clf.classes_)
    return float(proba[classes.index(1)]) if 1 in classes else 0.0


def has_model(session_id: str, agent_id: str) -> bool:
    return (session_id, agent_id) in _MODELS


# ── Training ──────────────────────────────────────────────────────────────────

def _export_tree(clf, labels_by_leaf: dict) -> dict:
    """
    Walk the fitted sklearn tree into a nested dict the UI can draw.

    Internal nodes carry their split (`PC{k} <= t`) — the features are PCA components,
    so the condition isn't human-meaningful on its own, but the STRUCTURE, purity and
    sample counts are. Leaves carry the LLM-generated topic label.
    """
    t = clf.tree_
    classes = list(clf.classes_)

    def node(i: int) -> dict:
        counts = t.value[i][0]
        total = float(counts.sum()) or 1.0
        p_engage = float(counts[classes.index(1)] / total) if 1 in classes else 0.0
        ent = 0.0
        for p in (p_engage, 1.0 - p_engage):
            if p > 0:
                ent -= p * math.log2(p)
        left, right = int(t.children_left[i]), int(t.children_right[i])
        base = {
            "n": int(t.n_node_samples[i]),
            "engage_prob": round(p_engage, 3),
            "entropy": round(ent, 3),
        }
        if left == -1:                       # leaf
            base.update({
                "kind": "leaf",
                "label": labels_by_leaf.get(i, ""),
                "children": [],
            })
        else:
            base.update({
                "kind": "split",
                "feature": int(t.feature[i]),
                "threshold": round(float(t.threshold[i]), 4),
                "children": [node(left), node(right)],
            })
        return base

    return node(0)


def _entropy(labels: list) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    p1 = sum(labels) / n
    ent = 0.0
    for p in (p1, 1 - p1):
        if p > 0:
            ent -= p * math.log2(p)
    return round(ent, 3)


async def train_session(db: Session, session_id: str, tick: int, label_leaves: bool = True) -> dict:
    """Refit PCA + per-agent decision trees from the persisted clickstream, update the
    live model cache, and persist a human-readable snapshot per agent."""
    events = db.query(ClickstreamEvent).filter(
        ClickstreamEvent.session_id == session_id
    ).all()
    if not events:
        return {"agents_trained": 0}

    emb = {pe.post_id: pe.vector for pe in db.query(PostEmbedding).filter(
        PostEmbedding.session_id == session_id).all()}
    posts = {p.id: p for p in db.query(DeliberationPost).filter(
        DeliberationPost.session_id == session_id).all()}
    if not emb:
        return {"agents_trained": 0}

    all_pids = [pid for pid in emb]
    X_all = np.asarray([emb[pid] for pid in all_pids], dtype=float)

    # PCA on all observed post embeddings for speed; skip if too few samples.
    pca = None
    if X_all.shape[0] > 3:
        n_comp = min(PCA_COMPONENTS, X_all.shape[0] - 1, X_all.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=0).fit(X_all)

    def reduce(v):
        return pca.transform([v])[0] if pca is not None else np.asarray(v, dtype=float)

    reduced = {pid: reduce(emb[pid]) for pid in all_pids}
    gts = load_ground_truth(db, session_id)

    by_agent = defaultdict(list)
    for e in events:
        if e.post_id in emb:
            by_agent[e.agent_id].append(e)

    trained = 0
    for agent_id, evs in by_agent.items():
        if len(evs) < MIN_EVENTS_TO_TRAIN:
            continue
        X = np.asarray([reduced[e.post_id] for e in evs])
        y = np.asarray([int(e.label) for e in evs])
        gt = gts.get(agent_id)

        # Accuracy of the model at recovering the ground-truth boundary over all posts.
        def _accuracy(predict_fn) -> Optional[float]:
            if not gt:
                return None
            ytrue = np.asarray([1 if gt_engage(gt, emb[pid]) else 0 for pid in all_pids])
            ypred = np.asarray([predict_fn(reduced[pid]) for pid in all_pids])
            return round(float((ypred == ytrue).mean()), 4)

        if len(set(y.tolist())) < 2:
            const = int(y[0])
            _MODELS[(session_id, agent_id)] = {"pca": pca, "clf": None, "const": const}
            acc = _accuracy(lambda x: const)
            only = {"kind": "leaf", "label": "all " + ("engaged" if const else "skipped"),
                    "n": len(evs), "engage_prob": float(const), "entropy": 0.0, "children": []}
            _upsert_snapshot(db, session_id, agent_id, tick, len(evs), acc, 1.0 if acc else None,
                             leaves=[{**only, "leaf_id": 0, "examples": []}], tree_json=only)
            trained += 1
            continue

        clf = DecisionTreeClassifier(
            criterion="entropy", min_samples_leaf=MIN_SAMPLES_LEAF, random_state=0
        ).fit(X, y)
        _MODELS[(session_id, agent_id)] = {"pca": pca, "clf": clf, "const": None}

        train_acc = round(float((clf.predict(X) == y).mean()), 4)
        acc = _accuracy(lambda x: int(clf.predict([x])[0]))

        # Leaves = data-discovered subtopics.
        leaf_ids = clf.apply(X)
        by_leaf = defaultdict(list)
        for e, lid in zip(evs, leaf_ids):
            by_leaf[int(lid)].append(e)
        leaves = []
        for lid, lev in by_leaf.items():
            labels = [int(e.label) for e in lev]
            names = [posts[e.post_id].name for e in lev if e.post_id in posts][:6]
            leaves.append({
                "leaf_id": lid,
                "label": "",                      # filled by LLM below
                "n": len(lev),
                "engage_prob": round(sum(labels) / len(labels), 3),
                "entropy": _entropy(labels),
                "examples": names,
            })
        leaves.sort(key=lambda d: -d["n"])

        if label_leaves and leaves:
            names_map = await _label_leaves([lf["examples"] for lf in leaves])
            for lf, name in zip(leaves, names_map):
                lf["label"] = name

        # Preserve any labels from a previous (labelled) train so mid-run retrains,
        # which skip the LLM pass, don't blank out the names already shown in the UI.
        labels_by_leaf = {lf["leaf_id"]: lf["label"] for lf in leaves if lf.get("label")}
        if not label_leaves:
            prev = db.query(UserModelSnapshot).filter(
                UserModelSnapshot.session_id == session_id,
                UserModelSnapshot.agent_id == agent_id,
            ).first()
            if prev and prev.leaves:
                for lf in leaves:
                    if lf["label"]:
                        continue
                    old = next((o for o in prev.leaves if o.get("leaf_id") == lf["leaf_id"]), None)
                    if old and old.get("label"):
                        lf["label"] = old["label"]
                        labels_by_leaf[lf["leaf_id"]] = old["label"]

        tree_json = _export_tree(clf, labels_by_leaf)
        _upsert_snapshot(db, session_id, agent_id, tick, len(evs), acc, train_acc, leaves, tree_json)
        trained += 1

    db.commit()
    return {"agents_trained": trained}


def _upsert_snapshot(db, session_id, agent_id, tick, n_events, accuracy, train_accuracy,
                     leaves, tree_json=None):
    row = db.query(UserModelSnapshot).filter(
        UserModelSnapshot.session_id == session_id,
        UserModelSnapshot.agent_id == agent_id,
    ).first()
    if row:
        row.updated_tick = tick
        row.n_events = n_events
        row.accuracy = accuracy
        row.train_accuracy = train_accuracy
        row.leaves = leaves
        row.tree_json = tree_json
    else:
        db.add(UserModelSnapshot(
            session_id=session_id, agent_id=agent_id, updated_tick=tick,
            n_events=n_events, accuracy=accuracy, train_accuracy=train_accuracy,
            leaves=leaves, tree_json=tree_json,
        ))


async def _label_leaves(leaf_examples: list) -> list:
    """One LLM call: turn each leaf's example post names into a short topic label."""
    blocks = []
    for i, names in enumerate(leaf_examples):
        joined = "; ".join(names) if names else "(no examples)"
        blocks.append(f"{i}. {joined}")
    prompt = (
        "Each numbered group below is a cluster of deliberation posts that a participant "
        "treated the same way. Give each cluster a SHORT topic label (2-5 words) that "
        "captures what the posts have in common.\n\n"
        + "\n".join(blocks)
        + "\n\nReturn ONLY valid JSON, labels in the same order:\n"
        '{"labels": ["...", "..."]}'
    )
    try:
        resp = await _get_client().aio.models.generate_content(
            model=LABEL_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=600, temperature=0.4,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = re.sub(r"```(?:json)?", "", (resp.text or "").strip()).strip("`").strip()
        labels = json.loads(raw).get("labels", [])
    except Exception:
        labels = []
    out = []
    for i in range(len(leaf_examples)):
        out.append(str(labels[i]).strip()[:60] if i < len(labels) and labels[i] else f"cluster {i}")
    return out
