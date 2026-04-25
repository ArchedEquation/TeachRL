"""
app.py — TeachRL v2 Main Entry Point.

Tries to use OpenEnv create_app first (proper spec compliance).
Falls back to custom FastAPI if openenv-core not available in environment.
Port 7860 for HuggingFace Spaces.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from openenv.core.env_server.http_server import create_app
    from server.teachrl_environment import TeachRLEnvironment
    from models import TutorAction, TutorObservation

    app = create_app(
        TeachRLEnvironment,
        TutorAction,
        TutorObservation,
        env_name="teachrl",
        max_concurrent_envs=10,
    )
    _using_openenv = True

except ImportError:
    # openenv-core not available — use custom FastAPI (full featured)
    import uuid
    from typing import Dict, Optional
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
    from env.environment import TASK_REGISTRY
    from env.archetypes import CONCEPTS, DIFFICULTY_LEVELS, ALL_ARCHETYPES

    try:
        from server.teachrl_environment import TeachRLEnvironment
        from models import TutorAction as _TutorAction
    except Exception:
        from env.environment import TeachRLEnv as TeachRLEnvironment
        from env.environment import TutorAction as _TutorAction

    SESSIONS: Dict[str, TeachRLEnvironment] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        SESSIONS.clear()

    app = FastAPI(
        title="TeachRL-v2",
        description=(
            "OpenEnv Adaptive Tutoring — 8 Hidden Student Archetypes + "
            "Self-Play Escalation. Theme 4: Self-Improvement."
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

    def _get(sid):
        if sid not in SESSIONS:
            raise HTTPException(404, f"Session '{sid}' not found. Call /reset first.")
        return SESSIONS[sid]

    @app.get("/")
    def root():
        return {
            "env_id": "TeachRL-v2", "version": "2.0.0",
            "tasks": list(TASK_REGISTRY),
            "archetypes": [a.value for a in ALL_ARCHETYPES],
            "concepts": CONCEPTS, "difficulties": DIFFICULTY_LEVELS,
            "api": ["/reset", "/step", "/state", "/render", "/docs"],
            "status": "ok",
        }

    @app.post("/reset")
    def reset(req: Optional[ResetRequest] = None):
        if req is None: req = ResetRequest()
        if req.task_id not in TASK_REGISTRY:
            raise HTTPException(400, f"Unknown task: {req.task_id}")
        sid = str(uuid.uuid4())
        env = TeachRLEnvironment(task_id=req.task_id, seed=req.seed, eval_mode=req.eval_mode)
        obs = env.reset(seed=req.seed)
        SESSIONS[sid] = env
        return {
            "session_id": sid, "observation": obs.model_dump(),
            "task": {"id": req.task_id,
                     "difficulty": TASK_REGISTRY[req.task_id]["difficulty"],
                     "max_steps": TASK_REGISTRY[req.task_id]["max_steps"],
                     "description": TASK_REGISTRY[req.task_id]["description"]},
        }

    @app.post("/step")
    def step(req: StepRequest):
        env = _get(req.session_id)
        if env._done: raise HTTPException(400, "Episode done. Call /reset.")
        try:
            action = _TutorAction(concept=req.concept, difficulty=req.difficulty,
                                  hint_given=req.hint_given, archetype_guess=req.archetype_guess)
        except Exception as e:
            raise HTTPException(422, str(e))
        result = env.step(action)
        return {"observation": result.model_dump(), "reward": result.reward,
                "done": result.done, "info": result.metadata}

    @app.get("/state")
    def state(session_id: str = Query(...)):
        s = _get(session_id).state
        return s.model_dump() if hasattr(s, 'model_dump') else vars(s)

    @app.get("/render")
    def render(session_id: str = Query(...)):
        env = _get(session_id)
        return {"render": env.render() if hasattr(env, 'render') else "render not available"}

    @app.get("/tasks")
    def tasks():
        return {tid: {"difficulty": cfg["difficulty"], "description": cfg["description"],
                      "max_steps": cfg["max_steps"]} for tid, cfg in TASK_REGISTRY.items()}

    @app.get("/archetypes")
    def archetypes():
        from env.archetypes import ARCHETYPE_CLASSES
        return {aid.value: cls.__dict__.get("description","") for aid,cls in ARCHETYPE_CLASSES.items()}

    @app.delete("/session/{session_id}")
    def delete_session(session_id: str):
        if session_id not in SESSIONS: raise HTTPException(404, "Session not found")
        del SESSIONS[session_id]; return {"deleted": session_id}

    _using_openenv = False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)