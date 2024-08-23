import logging

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Uso en otros módulos
logger = setup_logger('main_logger', 'logs/main.log')
error_logger = setup_logger('error_logger', 'logs/error.log', level=logging.ERROR)
