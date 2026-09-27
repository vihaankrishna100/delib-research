import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from database import engine, Base, run_migrations
from routes import map, simulation, analytics
import seed as seed_module

load_dotenv()

Base.metadata.create_all(bind=engine)
run_migrations()
seed_module.seed()

app = FastAPI(title="Deliberation Map Simulation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(map.router)
app.include_router(simulation.router)
app.include_router(analytics.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agents")
def list_agents():
    from database import SessionLocal
    from models import Agent
    db = SessionLocal()
    try:
        agents = db.query(Agent).all()
        return [
            {
                "id": a.id,
                "name": a.name,
                "role": a.role.value,
                "domain": a.domain,
                "avatar_color": a.avatar_color,
                "llm_assist_rate": a.llm_assist_rate,
                "model_id": a.model_id,
                "is_active": a.is_active,
            }
            for a in agents
        ]
    finally:
        db.close()
