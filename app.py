from fastapi import FastAPI
import datetime

app = FastAPI()

@app.get("/hello")
async def hello():
    timestamp = datetime.datetime.now().isoformat()
    return f"Hello, World! - Test {timestamp}"
