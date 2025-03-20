import os
import logging
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("RDAGENT_LOG_LEVEL", "INFO")
log_dir = os.getenv("RDAGENT_LOG_DIR", "./logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/rdagent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rdagent")

# Create FastAPI app
app = FastAPI(title="RD Agent API", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "RD Agent API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("RDAGENT_HOST", "0.0.0.0")
    port = int(os.getenv("RDAGENT_PORT", "8000"))
    logger.info(f"Starting RD Agent API on {host}:{port}")
    uvicorn.run("rdagent.main:app", host=host, port=port, reload=True) 