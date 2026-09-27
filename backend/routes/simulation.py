import asyncio
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from database import get_db
from models import SimulationSession, SessionStatus, DeliberationPost, PostType
from schemas import SessionCreate, SessionOut, SimulationConfig

router = APIRouter(prefix="/simulation", tags=["simulation"])

# Track running tasks per session
_running_tasks: dict[str, asyncio.Task] = {}


@router.post("/sessions", response_model=SessionOut)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    # Create session
    session = SimulationSession(
        status=SessionStatus.idle,
        config=body.config.model_dump(),
    )
    db.add(session)
    db.flush()

    # Create mission post
    mission = DeliberationPost(
        session_id=session.id,
        ptype=PostType.mission,
        name=body.mission_name,
        description=body.mission_description,
        purpose_id=None,
        author_agent_id=None,
        content_mode="self_written",
        sim_tick=0,
    )
    db.add(mission)
    db.flush()

    session.mission_id = mission.id
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(db: Session = Depends(get_db)):
    return db.query(SimulationSession).order_by(SimulationSession.started_at.desc()).all()


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/sessions/{session_id}/start")
def start_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    ticks: int = 20,
    db: Session = Depends(get_db),
):
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status == SessionStatus.running:
        raise HTTPException(status_code=400, detail="Session already running")

    session.status = SessionStatus.running
    session.started_at = session.started_at or datetime.utcnow()
    db.commit()

    background_tasks.add_task(_run_sim_background, session_id, ticks)
    return {"status": "started", "session_id": session_id}


@router.post("/sessions/{session_id}/pause")
def pause_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.status = SessionStatus.paused
    db.commit()
    return {"status": "paused"}


@router.post("/sessions/{session_id}/resume")
def resume_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    ticks: int = 20,
    db: Session = Depends(get_db),
):
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status == SessionStatus.running:
        raise HTTPException(status_code=400, detail="Already running")

    session.status = SessionStatus.running
    db.commit()
    background_tasks.add_task(_run_sim_background, session_id, ticks)
    return {"status": "resumed"}


@router.post("/sessions/{session_id}/stop")
def stop_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.status = SessionStatus.stopped
    session.ended_at = datetime.utcnow()
    db.commit()
    return {"status": "stopped"}


def _run_sim_background(session_id: str, ticks: int):
    """Runs simulation in FastAPI background task."""
    import asyncio
    from database import SessionLocal
    from simulation.engine import run_simulation

    db = SessionLocal()
    try:
        asyncio.run(run_simulation(db, session_id, ticks))
    finally:
        db.close()
