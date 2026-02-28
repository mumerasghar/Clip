import os
import logging

def setup_logger(log_file_path: str | os.PathLike):
    """
    Sets up a logger that logs to both the console and a file.

    :param log_file_path: Path to the file where logs should be saved.
    """
    # Configure the logging system
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to debug
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Set the format of the log messages
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),  # Log to a file at the specified path
            logging.StreamHandler()  # Log to the console
        ]
    )

    # Create a logger instance
    logger = logging.getLogger('Test-Task-Logger')

    return logger

