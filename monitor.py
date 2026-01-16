import threading
import time
import csv
import os
import psutil
import datetime
import server
import gc
import torch
import collections
import inspect
import sys
import hashlib
from folder_paths import get_output_directory

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

try:
    import comfy.model_management
    COMFY_MM_AVAILABLE = True
except ImportError:
    COMFY_MM_AVAILABLE = False

try:
    from comfy.isolation.model_patcher_proxy_registry import ModelPatcherRegistry
    from comfy.isolation.clip_proxy import CLIPRegistry
    from comfy.isolation.vae_proxy import VAERegistry
    from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry
    ISOLATION_AVAILABLE = True
except ImportError:
    ISOLATION_AVAILABLE = False

# --- Configuration ---
POLLING_RATE = 0.1 # 10Hz
# Allow override for unique run artifacts (Battery Test Automation)
filename = os.environ.get("STABILITY_METRICS_TARGET", "stability_metrics.csv")
OUTPUT_FILE = os.path.join(get_output_directory(), filename)
EVENTS_TO_LOG = {
    "execution_start": "START",
    "execution_success": "SUCCESS",
    "execution_error": "ERROR",
    "execution_interrupted": "INTERRUPTED",
    # "executing": "NODE_START" # Optional: enable for node-level granularity
}

class CGPUInfo:
    def __init__(self):
        self.pynvml_loaded = False
        self.handle = None
        if pynvml_available:
            try:
                pynvml.nvmlInit()
                self.pynvml_loaded = True
                if pynvml.nvmlDeviceGetCount() > 0:
                    self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Monitor primary GPU
            except Exception as e:
                print(f"[ComfyUI-StabilityTest] GPU Init Failed: {e}")

    def get_status(self):
        vram_used = 0
        vram_total = 0
        gpu_util = 0
        vram_host = 0
        vram_child = 0
        
        if self.pynvml_loaded and self.handle:
            try:
                # 1. Global Memory
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                vram_used = mem.used
                vram_total = mem.total
                
                # 2. Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    gpu_util = util.gpu
                except:
                    gpu_util = 0

                # 3. Process-Level VRAM (Compute + Graphics)
                # pynvml might separate compute/graphics. We check both or specific one.
                # Usually Compute is enough for CUDA, but let's be safe.
                # Note: This is an expensive call, hopefully <1ms.
                procs = []
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                except:
                    pass
                
                # Fallback or additive? Usually specific to context.
                # Graphics processes needed?
                try:
                    g_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)
                    procs.extend(g_procs)
                except:
                    pass

                my_pid = os.getpid()
                # Find children once per call? Or rely on simple PID matching?
                # psutil is expensive. We assume any other python process on GPU is likely us in this isolated test env.
                # A safer check: Is the PID a child of ours?
                current_process = psutil.Process()
                children = [p.pid for p in current_process.children(recursive=True)]
                
                for p in procs:
                    if p.pid == my_pid:
                        vram_host += p.usedGpuMemory if p.usedGpuMemory else 0
                    elif p.pid in children:
                        vram_child += p.usedGpuMemory if p.usedGpuMemory else 0

            except Exception:
                pass
        return vram_used, vram_total, gpu_util, vram_host, vram_child

class CMonitor:
    def __init__(self):
        self.gpu_info = CGPUInfo()
        self.stop_event = threading.Event()
        self.thread = None
        self.file_handle = None
        self.writer = None
        self.lock = threading.Lock()
        
        # Provenance Tracking
        self.workflow_counter = 0
        self.tensor_registry = {} # {id(obj): origin_workflow_idx}
        
        # Lifecycle Logging
        self.lifecycle_log_path = os.path.join(get_output_directory(), "model_lifecycle.jsonl")
        # Clear previous log
        if os.path.exists(self.lifecycle_log_path):
             try:
                 os.remove(self.lifecycle_log_path)
             except:
                 pass

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        
        # Initialize CSV
        is_new_file = not os.path.exists(OUTPUT_FILE)
        self.file_handle = open(OUTPUT_FILE, 'a', newline='')
        self.writer = csv.writer(self.file_handle)
        
        if is_new_file:
            self.writer.writerow(["timestamp", "event_type", "vram_used_bytes", "vram_total_bytes", "ram_used_bytes", "ram_total_bytes", "gpu_util_percent", "cpu_util_percent", "vram_host_bytes", "vram_child_bytes"])
            self.file_handle.flush()

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        # Patch Server
        self._patch_server()

    def _patch_server(self):
        if not hasattr(server.PromptServer, "original_send_sync"):
            server.PromptServer.original_send_sync = server.PromptServer.send_sync
            
            def hooked_send_sync(instance, event, data, sid=None):
                if event in EVENTS_TO_LOG:
                    self.log_marker(EVENTS_TO_LOG[event])
                return server.PromptServer.original_send_sync(instance, event, data, sid)
                
            server.PromptServer.send_sync = hooked_send_sync
            print("\033[34m[ComfyUI-StabilityTest] \033[0mServer Patched for Event Sync")

    def log_marker(self, marker_name):
        # Update State
        if marker_name == "START":
            self.workflow_counter += 1

        # Perform Tensor Census on Start/Success/Error
        census_summary = "{}"
        if marker_name in ["START", "SUCCESS", "ERROR", "INTERRUPTED"]:
             # census_summary = self.perform_tensor_census()
             pass

        self.log_census_to_file(marker_name, census_summary)

        # Log simple marker to CSV for backward compatibility
        if self.writer:
            with self.lock:
                try:
                    now = datetime.datetime.now().isoformat()
                    # Use 0s to indicate marker row
                    self.writer.writerow([now, f"MARKER_{marker_name}", 0, 0, 0, 0, 0, 0, 0, 0])
                    self.file_handle.flush()
                except Exception as e:
                    print(f"[Monitor] Error logging marker: {e}")

    def log_census_to_file(self, marker, summary):
        log_path = os.path.join(os.path.dirname(OUTPUT_FILE), "tensor_audit.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": marker,
            "workflow_idx": self.workflow_counter,
            "census": summary
        }
        try:
            with open(log_path, "a") as f:
                import json
                f.write(json.dumps(entry) + "\n")
        except:
            pass

    def _hash_tensor(self, t):
        try:
            # Simple strided hash for performance
            # We want to identify content identity.
            if t.numel() == 0:
                return "empty"
            
            # Use data_ptr for identity within process, but for cross-run we need content.
            # We can't use data_ptr across runs.
            # We'll grab a sample.
            flat = t.flatten()
            len_t = flat.shape[0]
            
            # Sample: Start, Mid, End
            indices = [0, len_t // 2, len_t - 1]
            samples = []
            for i in indices:
                if i < len_t:
                    samples.append(flat[i].item())
            
            # Also include sum/mean for robustness if cheap?
            # On GPU sync is expensive.
            # Let's try to just use the sample tuple as the hash string.
            return f"{samples}"
        except:
            return "err"


    def perform_tensor_census(self):
        try:
            # STRICTLY NO GC.COLLECT() HERE
            # We observe the state as is.
            
            objects = collections.Counter() # Key: "(Shape)_Dtype_Origin_Hash_Ref"
            total_mb = 0
    
            def inspect_referrers(obj, depth=0):
                if depth > 1:
                    return "..."
                try:
                    refs = gc.get_referrers(obj)
                except Exception as e:
                    return [f"<Error getting referrers: {e}>"]

                ref_info = []
                for r in refs:
                    if r is obj: continue # Self ref
                    if hasattr(r, "__code__") and r.__code__ == inspect.currentframe().f_code: continue # Local frame
                    
                    try:
                        r_type = type(r).__name__
                        r_str = str(r)[:100]
                        if isinstance(r, list):
                            r_str = f"List(len={len(r)})"
                        elif isinstance(r, dict):
                            r_str = f"Dict(len={len(r)}, keys={list(r.keys())[:5]})"
                        
                        ref_info.append(f"{r_type}: {r_str}")
                    except Exception as e:
                        ref_info.append(f"<Error inspect ref: {e}>")
                        
                return ref_info

            # Perform census
            # -------------------------------------------------------------
            current_tensors = {}
            provenance_breakdown = collections.defaultdict(lambda: collections.defaultdict(int))
            
            inspected_count = 0
            try:
                # We need to be careful not to keep references ourselves
                # so we iterate and process immediately
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            # Metrics
                            obj_size_mb = (obj.element_size() * obj.nelement()) / (1024*1024)
                            total_mb += obj_size_mb

                            # 1. Identity
                            oid = id(obj)
                            current_tensors[oid] = True # Mark as alive

                            # 2. Registration (Provenance)
                            # If meaningful new tensor, register it
                            if oid not in self.tensor_registry:
                                # Use current workflow counter as origin
                                self.tensor_registry[oid] = self.workflow_counter

                            origin_workflow_idx = self.tensor_registry[oid]
                            
                            # 3. Hash & RefCount
                            # Lightweight Identity Hash
                            thash = self._hash_tensor(obj)
                            
                            # RefCount
                            # Note: getrefcount returns +1 (temp ref)
                            ref_count = sys.getrefcount(obj) - 1

                            # Log Referrers for suspected leaks (Ref >= 3 and from previous workflows)
                            # Only inspecting leaks from Workflow 1 (index 1) if we are in later workflow
                            # And ensure we don't spam (limit 5)
                            if ref_count >= 3:
                                 # print(f"DEBUG: HighRef OID={oid} Ref={ref_count} Origin={origin_workflow_idx} Counter={self.workflow_counter}")
                                 pass

                            if inspected_count < 5 and ref_count >= 3 and origin_workflow_idx == 1 and self.workflow_counter > 1:
                                 print(f"!!! SUSPECT TENSOK LEAK !!! OID={oid} RefCount={ref_count} Origin=W{origin_workflow_idx}")
                                 print(f"Referrers: {inspect_referrers(obj)}")
                                 inspected_count += 1

                            # 4. Aggregation
                            # Group by (Origin, Shape, Dtype)
                            key = f"{tuple(obj.shape)}_{obj.dtype}"
                            
                            # We just store counts by origin-type signature for the summary
                            # And building the detailed breakdown string for "objects" counter
                            
                            # Signature (tuple shape, dtype, origin, hash, ref)
                            sig = f"{tuple(obj.shape)}_{obj.dtype}_W{origin_workflow_idx}_H[{thash}]_Ref{ref_count}"
                            objects[sig] += 1
                            
                    except Exception:
                        continue  # obj might be dead
            except Exception as e:
                print(f"Error during tensor census: {e}")
                return {"error": str(e)}
            
            # 2. Prune Registry (Objects that died)
            # current_tensors keys are the live OIDs
            keys_to_remove = [k for k in self.tensor_registry if k not in current_tensors]
            for k in keys_to_remove:
                del self.tensor_registry[k]

            return {
                "total_vram_objects_mb": round(total_mb, 2),
                "breakdown": dict(objects)
            }

        except Exception as e:
            return {"error": str(e)}

    def _run(self):
        while not self.stop_event.is_set():
            now = datetime.datetime.now().isoformat()
            
            # Hardware Stats
            cpu_util = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            vram_used, vram_total, gpu_util, vram_host, vram_child = self.gpu_info.get_status()
            
            with self.lock:
                self.writer.writerow([
                    now, 
                    "TELEMETRY", 
                    vram_used, 
                    vram_total, 
                    ram.used, 
                    ram.total, 
                    gpu_util, 
                    cpu_util,
                    vram_host,
                    vram_child
                ])
                self.file_handle.flush()

            # Lifecycle Telemetry
            lifecycle_state = self.capture_lifecycle_state()
            self.log_lifecycle_to_file(lifecycle_state)
            
            time.sleep(POLLING_RATE)

    def capture_lifecycle_state(self):
        state = {
            "timestamp": datetime.datetime.now().isoformat(),
            "loaded_models": [],
            "registries": {}
        }
        
        # 1. ComfyUI Native Model Management
        if COMFY_MM_AVAILABLE:
            try:
                # Accessing global list directly as it's modified in place
                for loaded_model in comfy.model_management.current_loaded_models:
                    real_model = loaded_model.model
                    model_info = {
                        "type": type(real_model).__name__,
                        "device": str(loaded_model.device),
                        "model_size": loaded_model.model_size(),
                        # Check currently_used flag - this is the CRITICAL metric
                        "currently_used": getattr(loaded_model, "currently_used", "UNKNOWN")
                    }
                    state["loaded_models"].append(model_info)
            except Exception as e:
                state["loaded_models_error"] = str(e)

        # 2. PyIsolate Registries
        if ISOLATION_AVAILABLE:
            try:
                # We need to access the singleton instances safely
                # Note: These are WeakValueDictionaries now per stash@{1}, checking len() might be tricky 
                # or we just iterate. WeakValueDictionary has len().
                
                # Helper to safely get len
                def safe_len(registry_cls):
                    try:
                        # Assuming registry singleton pattern: registry_cls() returns instance
                        return len(registry_cls()._registry)
                    except:
                        return -1

                state["registries"]["ModelPatcher"] = safe_len(ModelPatcherRegistry)
                state["registries"]["CLIP"] = safe_len(CLIPRegistry)
                state["registries"]["VAE"] = safe_len(VAERegistry)
                state["registries"]["ModelSampling"] = safe_len(ModelSamplingRegistry)
            except Exception as e:
                state["registries_error"] = str(e)
                
        return state

    def log_lifecycle_to_file(self, state):
        try:
            with open(self.lifecycle_log_path, "a") as f:
                import json
                f.write(json.dumps(state) + "\n")
        except:
            pass

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        if self.file_handle:
            self.file_handle.close()

_monitor_instance = CMonitor()

def start_monitoring():
    _monitor_instance.start()
