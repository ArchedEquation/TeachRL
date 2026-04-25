"""
server/app.py — TeachRL v2 OpenEnv Server using create_app.

Uses openenv.core.env_server.http_server.create_app as required by the spec.
Exposes reset / step / state endpoints plus WebSocket for persistent sessions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
except ImportError:
    # Fallback: use our custom FastAPI app if openenv-core not available
    from app import app  # noqa: F401


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()