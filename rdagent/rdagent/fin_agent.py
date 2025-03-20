import logging
import os
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
        logging.FileHandler(f"{log_dir}/fin_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rdagent.fin_agent")

def main():
    logger.info("Starting fin_agent...")
    # Add your fin_agent logic here
    logger.info("fin_agent completed.")

if __name__ == "__main__":
    main() 