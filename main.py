import uvicorn
from fastapi.responses import JSONResponse
from app.graphql.schema import PlanManager
from app.process_manager import ProcessManager
from app.config import config
from app import app

plan_manager = PlanManager()

# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:

    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


if __name__ == "__main__":
    manager = ProcessManager(size=config.process_manager.size,
                             timeout=config.process_manager.timeout)
    uvicorn.run(app, host='127.0.0.1', port=8000)
