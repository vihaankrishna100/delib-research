from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from database import get_db
from models import DeliberationPost, PostType
from schemas import PostOut, PostWithAchievers

router = APIRouter(prefix="/map", tags=["map"])


def _post_to_dict(p: DeliberationPost) -> dict:
    return {
        "id": p.id,
        "session_id": p.session_id,
        "ptype": p.ptype.value,
        "name": p.name,
        "description": p.description,
        "purpose_id": p.purpose_id,
        "author_agent_id": p.author_agent_id,
        "content_mode": p.content_mode,
        "is_removed": p.is_removed,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "sim_tick": p.sim_tick,
    }


def _build_tree(post_dict: dict, all_posts: dict) -> dict:
    """Recursively build nested tree from flat dict."""
    node = dict(post_dict)
    node["achievers"] = [
        _build_tree(child, all_posts)
        for child in all_posts.values()
        if child.get("purpose_id") == post_dict["id"] and not child.get("is_removed")
    ]
    return node


@router.get("/tree")
def get_tree(session_id: str, db: Session = Depends(get_db)):
    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
        )
        .order_by(DeliberationPost.sim_tick.asc())
        .all()
    )
    if not posts:
        return {"roots": [], "all_posts": {}}

    all_posts = {p.id: _post_to_dict(p) for p in posts}
    # Any mission post with no purpose_id is a root
    root_posts = [p for p in all_posts.values() if p["ptype"] == "mission" and not p.get("purpose_id")]
    roots = [_build_tree(r, all_posts) for r in root_posts]
    return {"roots": roots, "all_posts": all_posts}


@router.get("/posts/{post_id}")
def get_post(post_id: str, db: Session = Depends(get_db)):
    post = db.query(DeliberationPost).filter(DeliberationPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return _post_to_dict(post)


@router.get("/posts/{post_id}/context")
def get_post_context(post_id: str, db: Session = Depends(get_db)):
    """Returns the chain from this post up to the mission root."""
    post = db.query(DeliberationPost).filter(DeliberationPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    chain = []
    current = post
    while current is not None:
        chain.append(_post_to_dict(current))
        pid = current.purpose_id
        current = db.query(DeliberationPost).filter(DeliberationPost.id == pid).first() if pid else None

    chain.reverse()
    return chain


@router.get("/posts")
def list_posts(
    session_id: str,
    ptype: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(DeliberationPost).filter(
        DeliberationPost.session_id == session_id,
        DeliberationPost.is_removed == False,
    )
    if ptype:
        try:
            q = q.filter(DeliberationPost.ptype == PostType(ptype))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid ptype: {ptype}")
    posts = q.order_by(DeliberationPost.sim_tick.asc()).all()
    return [_post_to_dict(p) for p in posts]
