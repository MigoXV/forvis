import logging

logger = logging.getLogger(__name__)

try:
    import dotenv

    dotenv.load_dotenv()
    logger.info("Loaded .env file")
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env loading")
