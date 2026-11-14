"""
Generate required deployment files in C:\\CARLA_0.9.16:
- Start_CARLA.bat
- Stop_CARLA.bat
- test_connection.py
- diagnostics.py
- README_TRAJPILOT_CARLA.md
"""

from pathlib import Path
import shutil

CARLA_DIR = Path(r"C:\CARLA_0.9.16")
PROJECT_ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow")

START_BAT = """@echo off
cd /d C:\\CARLA_0.9.16
echo ==============================================================================
echo Starting CARLA Simulator 0.9.16...
echo ==============================================================================
start "" CarlaUE4.exe -quality-level=Epic
"""

STOP_BAT = """@echo off
echo Stopping CARLA Simulator...
taskkill /f /im CarlaUE4.exe /t >nul 2>&1
echo CARLA simulator stopped.
"""

TEST_CONN = '''import sys
try:
    import carla
    print("=" * 70)
    print("CARLA 0.9.16 CONNECTION TEST")
    print("=" * 70)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    server_ver = client.get_server_version()
    world = client.get_world()
    settings = world.get_settings()
    print(f"CARLA API Version    : 0.9.16")
    print(f"CARLA Server Version : {server_ver}")
    print(f"Active Map           : {world.get_map().name}")
    print(f"Synchronous Mode     : {settings.synchronous_mode}")
    print(f"Fixed Delta Seconds  : {settings.fixed_delta_seconds}")
    print(f"Active Actors        : {len(world.get_actors())}")
    print("=" * 70)
    print("CONNECTION SUCCESSFUL")
    print("=" * 70)
except Exception as e:
    print("\\n[DIAGNOSTIC REPORT]")
    print(f"[CAUSE] Cannot connect to CARLA simulator on 127.0.0.1:2000 ({e})")
    print("[FIX]   The CARLA server process (CarlaUE4.exe) is not running.")
    print("[COMMAND] Run C:\\\\CARLA_0.9.16\\\\Start_CARLA.bat to launch simulator.")
    sys.exit(1)
'''

README_CONTENT = """# CARLA 0.9.16 - TrajPilot CityFlow Deployment

Installation Directory : C:\\CARLA_0.9.16
Environment Name       : carla0916 / carla_py312 (Python 3.12)
Project Workspace      : C:\\Users\\SOMUJIT\\Desktop\\cityflow

## Commands:
- Start CARLA   : Start_CARLA.bat
- Stop CARLA    : Stop_CARLA.bat
- Test Port     : python test_connection.py
- Diagnostics   : python diagnostics.py
- Run TrajPilot : C:\\Users\\SOMUJIT\\Desktop\\cityflow\\RUN_TRAJPILOT_CARLA.bat
"""

def generate():
    CARLA_DIR.mkdir(parents=True, exist_ok=True)
    
    (CARLA_DIR / "Start_CARLA.bat").write_text(START_BAT, encoding="ascii")
    (CARLA_DIR / "Stop_CARLA.bat").write_text(STOP_BAT, encoding="ascii")
    (CARLA_DIR / "test_connection.py").write_text(TEST_CONN, encoding="ascii")
    (CARLA_DIR / "README_TRAJPILOT_CARLA.md").write_text(README_CONTENT, encoding="ascii")
    
    # Copy diagnostics.py
    src_diag = PROJECT_ROOT / "diagnostics.py"
    if src_diag.exists():
        shutil.copyfile(src_diag, CARLA_DIR / "diagnostics.py")
        
    print("Successfully generated all files in C:\\CARLA_0.9.16:")
    for f in ["Start_CARLA.bat", "Stop_CARLA.bat", "test_connection.py", "diagnostics.py", "README_TRAJPILOT_CARLA.md"]:
        p = CARLA_DIR / f
        print(f"  - {f}: {'EXISTS' if p.exists() else 'MISSING'} ({p.stat().st_size} bytes)")

if __name__ == "__main__":
    generate()
