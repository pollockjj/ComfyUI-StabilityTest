from .monitor import start_monitoring

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("\033[34m[ComfyUI-StabilityTest] \033[0mLoading StabilityTest Monitor...")
start_monitoring()
print("\033[34m[ComfyUI-StabilityTest] \033[0mMonitoring Active (10Hz -> stability_metrics.csv)")
