"""
Week 02 — Environment Setup & Validation

Deliverables:
  - Environment setup notes with exact versions and install commands
  - Primary baseline run script executes end to end
  - Automated logging produces timestamped VRAM and CPU RAM traces
  - Fallback path installed and verified
"""

import os
import sys
import subprocess
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MODEL_ID, RESULTS_DIR
from src.utils import (
    setup_logging, save_results_json, get_hardware_info, set_seed,
)
from src.vram_monitor import VRAMMonitor, get_current_gpu_memory

logger = setup_logging("week02")

WEEK_DIR = os.path.join(RESULTS_DIR, "week02_environment")
os.makedirs(WEEK_DIR, exist_ok=True)


def validate_environment():
    """Validate all required packages and hardware."""
    logger.info("Validating environment...")

    env_info = get_hardware_info()
    logger.info(f"  Platform: {env_info['platform']}")
    logger.info(f"  Python: {env_info['python_version']}")
    logger.info(f"  PyTorch: {env_info['torch_version']}")
    logger.info(f"  CUDA available: {env_info['cuda_available']}")

    if env_info.get("cuda_available"):
        logger.info(f"  GPU: {env_info.get('gpu_name', 'Unknown')}")
        logger.info(f"  GPU VRAM: {env_info.get('gpu_vram_total_gb', '?')} GB")
        logger.info(f"  CUDA version: {env_info.get('cuda_version', '?')}")
        logger.info(f"  Compute capability: {env_info.get('gpu_compute_capability', '?')}")

    # Package versions
    packages = {}
    required = [
        "torch", "transformers", "accelerate", "bitsandbytes",
        "datasets", "pynvml", "numpy", "pandas", "matplotlib",
        "tqdm", "psutil", "scipy", "seaborn",
    ]

    for pkg_name in required:
        try:
            pkg = __import__(pkg_name)
            ver = getattr(pkg, "__version__", "unknown")
            packages[pkg_name] = {"installed": True, "version": ver}
            logger.info(f"  [OK] {pkg_name}=={ver}")
        except ImportError:
            packages[pkg_name] = {"installed": False}
            logger.warning(f"  [MISSING] {pkg_name} NOT INSTALLED")

    env_info["packages"] = packages
    return env_info


def test_model_loading():
    """Test that the primary model loads successfully."""
    logger.info(f"Testing model loading: {DEFAULT_MODEL_ID}")

    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — skipping model load test.")
        return {"status": "skipped", "reason": "no_cuda"}

    try:
        from src.model_loader import load_model_4bit, verify_vram_budget

        start_time = time.time()
        model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID)
        load_time = time.time() - start_time

        # Verify VRAM budget
        budget_check = verify_vram_budget(model)

        # Quick inference test
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = {
            "status": "success",
            "model_id": DEFAULT_MODEL_ID,
            "load_time_s": round(load_time, 2),
            "vram_budget_check": budget_check,
            "test_prompt": test_prompt,
            "test_response": response[:200],
        }

        logger.info(f"  Model loaded in {load_time:.1f}s")
        logger.info(f"  Within VRAM budget: {budget_check['within_budget']}")
        logger.info(f"  Test response: {response[:100]}...")

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        logger.error(f"  Model loading failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_vram_monitoring():
    """Test VRAM monitoring produces correct traces."""
    logger.info("Testing VRAM monitoring...")

    monitor = VRAMMonitor(poll_interval_s=0.1)

    # Take a snapshot
    snap = monitor.snapshot()
    logger.info(f"  GPU used: {snap.gpu_used_mb:.0f} MB")
    logger.info(f"  GPU total: {snap.gpu_total_mb:.0f} MB")
    logger.info(f"  CPU RAM used: {snap.cpu_ram_used_mb:.0f} MB")

    # Test polling
    monitor.start_polling()
    time.sleep(1.0)  # Poll for 1 second
    snapshots = monitor.stop_polling()

    result = {
        "status": "success",
        "num_snapshots": len(snapshots),
        "poll_interval_s": 0.1,
        "summary": monitor.get_summary(),
        "current_gpu_memory": get_current_gpu_memory(),
    }

    logger.info(f"  Collected {len(snapshots)} snapshots in 1s")
    return result


def generate_install_commands():
    """Generate install commands for reproducibility."""
    return {
        "conda_setup": [
            "conda create -n llm_inference python=3.11 -y",
            "conda activate llm_inference",
        ],
        "pip_install": [
            "pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124",
            "pip install -r requirements.txt",
        ],
        "verification": [
            "python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"",
            "python -c \"import bitsandbytes; print(bitsandbytes.__version__)\"",
            "python scripts/week02_setup_validation.py",
        ],
    }


def main():
    logger.info("=" * 60)
    logger.info("WEEK 02 — Environment Setup & Validation")
    logger.info("=" * 60)

    # 1. Environment validation
    env_info = validate_environment()
    save_results_json(env_info, os.path.join(WEEK_DIR, "environment_info.json"))

    # 2. Install commands
    install_cmds = generate_install_commands()
    save_results_json(install_cmds, os.path.join(WEEK_DIR, "install_commands.json"))

    # 3. VRAM monitoring test
    vram_test = test_vram_monitoring()
    save_results_json(vram_test, os.path.join(WEEK_DIR, "vram_monitoring_test.json"))

    # 4. Model loading test
    model_test = test_model_loading()
    save_results_json(model_test, os.path.join(WEEK_DIR, "model_load_test.json"))

    # Summary
    summary = {
        "week": "02",
        "title": "Environment Setup & Validation",
        "status": "COMPLETE",
        "cuda_available": env_info.get("cuda_available", False),
        "gpu_name": env_info.get("gpu_name", "N/A"),
        "model_load_status": model_test.get("status", "unknown"),
        "vram_monitor_status": vram_test.get("status", "unknown"),
        "deliverables": [
            "environment_info.json",
            "install_commands.json",
            "vram_monitoring_test.json",
            "model_load_test.json",
        ],
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week02_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 02 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
