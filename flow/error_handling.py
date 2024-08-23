import traceback
from logger import error_logger

def handle_error(e):
    error_message = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
    error_logger.error(error_message)
    # Aquí podrías enviar una alerta por email o notificación

# Uso
try:
    pass# Algún código que puede fallar
except Exception as e:
    handle_error(e)
