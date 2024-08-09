import os
import logging

def setup_logger(log_file_name, log_folder='logs'):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    log_path = os.path.join(log_folder, log_file_name)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(log_format)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
