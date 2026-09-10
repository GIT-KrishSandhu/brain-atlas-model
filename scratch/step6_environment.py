"""
STEP 6 — Computational environment audit.
Writes ENVIRONMENT_AUDIT.md to reports/data_survey/
READ-ONLY. Does not install or upgrade anything.
"""
import sys, platform, pathlib, subprocess, json

OUT = pathlib.Path(r"d:\NLP_Project\reports\data_survey")

lines = []
def p(s=""):
    lines.append(s)
    print(s)

p("=" * 60)
p("ENVIRONMENT AUDIT — TopAneu Brain Atlas Project")
p("=" * 60)

# Python
p(f"\nPython version:     {sys.version}")
p(f"Python executable:  {sys.executable}")
p(f"Platform:           {platform.platform()}")
p(f"Architecture:       {platform.machine()}")

# Key packages
pkgs = ["torch","torchvision","nibabel","numpy","scipy","matplotlib",
        "pandas","sklearn","monai","SimpleITK","nnunetv2","dicom2nifti"]
p("\n--- Python Package Versions ---")
pkg_info = {}
for pkg in pkgs:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        p(f"  {pkg:20s}: {ver}")
        pkg_info[pkg] = ver
    except ImportError:
        p(f"  {pkg:20s}: NOT INSTALLED")
        pkg_info[pkg] = "NOT_INSTALLED"

# CUDA / PyTorch
p("\n--- CUDA / GPU ---")
try:
    import torch
    p(f"  torch.cuda.is_available():    {torch.cuda.is_available()}")
    p(f"  torch.cuda.device_count():    {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name  = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / (1024**3)
            p(f"  Device {i}: {name}")
            p(f"    Total VRAM: {vram:.2f} GB")
            p(f"    CUDA capability: {props.major}.{props.minor}")
            p(f"    Multi-processors: {props.multi_processor_count}")
    p(f"  torch.__version__:            {torch.__version__}")
    p(f"  torch.version.cuda:           {torch.version.cuda}")
    p(f"  torch.backends.cudnn.version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
except ImportError:
    p("  PyTorch NOT INSTALLED")

# RAM
p("\n--- System RAM ---")
try:
    import psutil
    vm = psutil.virtual_memory()
    p(f"  Total RAM:      {vm.total/(1024**3):.2f} GB")
    p(f"  Available RAM:  {vm.available/(1024**3):.2f} GB")
    p(f"  Used RAM:       {vm.used/(1024**3):.2f} GB")
except ImportError:
    p("  psutil not installed — RAM info unavailable")

# Disk
p("\n--- Storage ---")
try:
    import shutil
    total, used, free = shutil.disk_usage(r"d:\\")
    p(f"  D:\\ Total:  {total/(1024**3):.1f} GB")
    p(f"  D:\\ Used:   {used/(1024**3):.1f} GB")
    p(f"  D:\\ Free:   {free/(1024**3):.1f} GB")
except Exception as e:
    p(f"  Error: {e}")

# NVIDIA driver (via nvidia-smi)
p("\n--- NVIDIA Driver (nvidia-smi) ---")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,cuda_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        p(f"  {result.stdout.strip()}")
    else:
        p(f"  nvidia-smi error: {result.stderr.strip()}")
except FileNotFoundError:
    p("  nvidia-smi not found on PATH")
except Exception as e:
    p(f"  Error: {e}")

p("\n" + "=" * 60)

# Write markdown
md = [
    "# Environment Audit",
    "## Computational Environment — Brain Atlas / TopAneu-26",
    f"**Generated:** 2026-09-08",
    "",
    "## Python",
    f"- **Version:** `{sys.version}`",
    f"- **Executable:** `{sys.executable}`",
    f"- **Platform:** `{platform.platform()}`",
    "",
    "## Python Package Inventory",
    "| Package | Version |",
    "|---|---|",
]
for pkg, ver in pkg_info.items():
    status = "✅" if ver != "NOT_INSTALLED" else "❌"
    md.append(f"| `{pkg}` | {status} {ver} |")

md += [
    "",
    "## GPU / CUDA",
]
try:
    import torch
    md.append(f"- **torch.cuda.is_available():** `{torch.cuda.is_available()}`")
    md.append(f"- **PyTorch version:** `{torch.__version__}`")
    md.append(f"- **CUDA version (torch):** `{torch.version.cuda}`")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / (1024**3)
            md.append(f"- **GPU {i}:** {torch.cuda.get_device_name(i)}")
            md.append(f"- **VRAM:** {vram:.2f} GB")
            md.append(f"- **CUDA capability:** {props.major}.{props.minor}")
    else:
        md.append("- **⚠️ CUDA NOT available via torch — training will be CPU only unless resolved**")
except:
    md.append("- PyTorch not installed")

with open(OUT/"ENVIRONMENT_AUDIT.md","w",encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\nWrote ENVIRONMENT_AUDIT.md")
