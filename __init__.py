import logging
from .monitor import start_monitoring

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

logger.info("[ComfyUI-StabilityTest] Loading StabilityTest Monitor...")
start_monitoring()
logger.info("[ComfyUI-StabilityTest] Monitoring Active (10Hz -> stability_metrics.csv)")
