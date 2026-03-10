import logging
import os
from logging.handlers import RotatingFileHandler

# Define log folder
LOG_DIR = os.path.join(os.getcwd(), "logs")  # will create 'logs' in project root
os.makedirs(LOG_DIR, exist_ok=True)  # create folder if it doesn't exist

# Define log file
log_file_path = os.path.join(LOG_DIR, "app.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            filename=log_file_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)

logging.info("Logger initialized successfully")