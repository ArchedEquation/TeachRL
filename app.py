"""app.py — FastAPI server for TeachRL v2 on HuggingFace Spaces."""
from __future__ import annotations
import uuid
from typing import Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from env.environment import TeachRLEnv, TutorAction, TASK_REGISTRY
from env.archetypes import CONCEPTS, DIFFICULTY_LEVELS, ALL_ARCHETYPES

SESSIONS: Dict[str, TeachRLEnv] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    SESSIONS.clear()

app = FastAPI(
    title="TeachRL-v2",
    description=(
        "OpenEnv Adaptive Tutoring Environment with 8 Hidden Student Archetypes "
        "and Self-Play Difficulty Escalation. Theme 4: Self-Improvement."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

class ResetRequest(BaseModel):
    task_id:   str  = "blind_teaching"
    seed:      int  = 42
    eval_mode: bool = False

class StepRequest(BaseModel):
    session_id:      str
    concept:         str
    difficulty:      str
    hint_given:      bool          = False
    archetype_guess: Optional[str] = None

def _get(sid: str) -> TeachRLEnv:
    if sid not in SESSIONS:
        raise HTTPException(404, f"Session '{sid}' not found. Call /reset first.")
    return SESSIONS[sid]

@app.get("/")
def root():
    return {
        "env_id":      "TeachRL-v2",
        "version":     "2.0.0",
        "theme":       "Theme 4: Self-Improvement",
        "tasks":       list(TASK_REGISTRY),
        "concepts":    CONCEPTS,
        "difficulties": DIFFICULTY_LEVELS,
        "archetypes":  [a.value for a in ALL_ARCHETYPES],
        "api":         ["/reset", "/step", "/state", "/render", "/docs"],
        "status":      "ok",
    }

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None: req = ResetRequest()
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task: {req.task_id}")
    sid = str(uuid.uuid4())
    env = TeachRLEnv(task_id=req.task_id, seed=req.seed, eval_mode=req.eval_mode)
    obs = env.reset(seed=req.seed)
    SESSIONS[sid] = env
    return {
        "session_id":  sid,
        "observation": obs.model_dump(),
        "task": {
            "id":          req.task_id,
            "difficulty":  TASK_REGISTRY[req.task_id]["difficulty"],
            "max_steps":   TASK_REGISTRY[req.task_id]["max_steps"],
            "description": TASK_REGISTRY[req.task_id]["description"],
        },
    }

@app.post("/step")
def step(req: StepRequest):
    env = _get(req.session_id)
    if env._done: raise HTTPException(400, "Episode done. Call /reset.")
    try:
        action = TutorAction(concept=req.concept, difficulty=req.difficulty,
                             hint_given=req.hint_given, archetype_guess=req.archetype_guess)
    except Exception as e:
        raise HTTPException(422, str(e))
    result = env.step(action)
    return {"observation": result.observation.model_dump(), "reward": result.reward,
            "done": result.done, "info": result.info}

@app.get("/state")
def state(session_id: str = Query(...)):
    return _get(session_id).state().model_dump()

@app.get("/render")
def render(session_id: str = Query(...)):
    return {"render": _get(session_id).render()}

@app.get("/action_space")
def action_space(): return TeachRLEnv().action_space

@app.get("/observation_space")
def observation_space(): return TeachRLEnv().observation_space

@app.get("/tasks")
def tasks():
    return {tid: {"difficulty": cfg["difficulty"], "description": cfg["description"],
                  "max_steps": cfg["max_steps"]} for tid, cfg in TASK_REGISTRY.items()}

@app.get("/archetypes")
def archetypes():
    from env.archetypes import ARCHETYPE_CLASSES
    return {aid.value: cls.description for aid, cls in ARCHETYPE_CLASSES.items()}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in SESSIONS: raise HTTPException(404, "Session not found")
    del SESSIONS[session_id]; return {"deleted": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
