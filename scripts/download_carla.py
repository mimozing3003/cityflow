"""
TrajPilot CityFlow - Automated CARLA 0.9.16 Downloader and Extractor
Handles smart resume from partial browser downloads and multi-threaded Range requests.
"""

import os
import sys
import time
import zipfile
import shutil
import urllib.request
import concurrent.futures
from pathlib import Path

CARLA_ZIP_URL = "https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip"
EXPECTED_SIZE = 7812252810  # Exact Content-Length from downloads.carlasim.com
INSTALL_DIR = Path(r"C:\CARLA_Install")
DEST_DIR = Path(r"C:\CARLA_0.9.16")
TARGET_ZIP = INSTALL_DIR / "CARLA_0.9.16.zip"
CRDOWNLOAD = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\Unconfirmed 944649.crdownload")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def check_url():
    print("[1/5] Verifying official CARLA 0.9.16 download URL...")
    req = urllib.request.Request(CARLA_ZIP_URL, headers=HEADERS, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            length = int(resp.headers.get("Content-Length", 0))
            content_type = resp.headers.get("Content-Type", "")
            print(f"      HTTP Status  : {status}")
            print(f"      Remote Size  : {length:,} bytes ({length / (1024**3):.2f} GB)")
            print(f"      Content-Type : {content_type}")
            if status != 200 or length != EXPECTED_SIZE:
                raise RuntimeError(f"Unexpected response: status={status}, size={length}")
            print("      URL verified successfully.")
            return length
    except Exception as e:
        print(f"      ERROR verifying URL: {e}")
        raise


def prepare_zip_file():
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    
    if TARGET_ZIP.exists():
        current_size = TARGET_ZIP.stat().st_size
        print(f"[2/5] Found existing {TARGET_ZIP}: {current_size:,} bytes")
        if current_size == EXPECTED_SIZE:
            print("      Existing file matches expected size exactly.")
            return current_size
    else:
        current_size = 0

    if current_size < 6362274489 and CRDOWNLOAD.exists():
        cr_size = CRDOWNLOAD.stat().st_size
        print(f"[2/5] Discovered partial download {CRDOWNLOAD.name} ({cr_size:,} bytes, {cr_size / (1024**3):.2f} GB)")
        print("      Copying partial download to C:\\CARLA_Install\\CARLA_0.9.16.zip...")
        t0 = time.time()
        with open(CRDOWNLOAD, "rb") as src, open(TARGET_ZIP, "wb") as dst:
            shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
        print(f"      Copied in {time.time() - t0:.1f}s. Resuming from {TARGET_ZIP.stat().st_size:,} bytes.")
        current_size = TARGET_ZIP.stat().st_size
    return current_size


def download_remaining(current_size):
    if current_size >= EXPECTED_SIZE:
        print("[3/5] Download already 100% complete.")
        return

    remaining = EXPECTED_SIZE - current_size
    print(f"[3/5] Resuming download of remaining {remaining:,} bytes ({remaining / (1024**3):.2f} GB)...")
    
    chunk_size = 16 * 1024 * 1024  # 16 MB chunks
    chunks = []
    ptr = current_size
    while ptr < EXPECTED_SIZE:
        end = min(ptr + chunk_size - 1, EXPECTED_SIZE - 1)
        chunks.append((ptr, end))
        ptr = end + 1

    print(f"      Split into {len(chunks)} chunks of up to 16 MB each.")
    
    downloaded_bytes = 0
    start_time = time.time()
    
    # We download chunks sequentially or with small pool and write to file
    with open(TARGET_ZIP, "r+b" if TARGET_ZIP.exists() else "wb") as f:
        f.seek(current_size)
        
        for idx, (start_byte, end_byte) in enumerate(chunks, 1):
            expected_chunk_len = end_byte - start_byte + 1
            retries = 5
            chunk_data = None
            
            while retries > 0:
                try:
                    req_headers = dict(HEADERS)
                    req_headers["Range"] = f"bytes={start_byte}-{end_byte}"
                    req = urllib.request.Request(CARLA_ZIP_URL, headers=req_headers)
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        chunk_data = resp.read()
                        if len(chunk_data) == expected_chunk_len:
                            break
                        else:
                            retries -= 1
                            time.sleep(1)
                except Exception as ex:
                    retries -= 1
                    time.sleep(2)
            
            if chunk_data is None or len(chunk_data) != expected_chunk_len:
                raise IOError(f"Failed to download chunk {idx} ({start_byte}-{end_byte}) after retries.")
            
            f.seek(start_byte)
            f.write(chunk_data)
            f.flush()
            downloaded_bytes += len(chunk_data)
            
            elapsed = time.time() - start_time
            speed = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            pct = ((current_size + downloaded_bytes) / EXPECTED_SIZE) * 100
            print(f"      Progress: {pct:6.2f}% | Chunk {idx:2d}/{len(chunks)} | Speed: {speed:5.2f} MB/s | Total: {(current_size + downloaded_bytes)/(1024**3):.2f}/{EXPECTED_SIZE/(1024**3):.2f} GB", end="\r")
            
    print(f"\n      Download finished in {time.time() - start_time:.1f}s.")


def verify_archive():
    print("[4/5] Verifying ZIP integrity...")
    actual_size = TARGET_ZIP.stat().st_size
    print(f"      File size: {actual_size:,} bytes")
    if actual_size != EXPECTED_SIZE:
        raise ValueError(f"Size mismatch: expected {EXPECTED_SIZE}, got {actual_size}")
    
    with zipfile.ZipFile(TARGET_ZIP, "r") as zf:
        namelist = zf.namelist()
        print(f"      ZIP contains {len(namelist)} entries.")
        has_carla_exe = any("CarlaUE4.exe" in name for name in namelist)
        if not has_carla_exe:
            raise RuntimeError("Corrupt archive: CarlaUE4.exe not found in zip archive!")
        print("      Found CarlaUE4.exe inside archive.")
        print("      Testing archive CRC headers...")
        first_bad = zf.testzip()
        if first_bad is not None:
            raise RuntimeError(f"ZIP CRC corrupted at file: {first_bad}")
        print("      Archive integrity check PASSED.")


def extract_carla():
    print(f"[5/5] Extracting CARLA to {DEST_DIR}...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    exe_path = DEST_DIR / "CarlaUE4.exe"
    if exe_path.exists():
        print(f"      CarlaUE4.exe already exists at {exe_path}. Skipping full extraction.")
        return
        
    t0 = time.time()
    with zipfile.ZipFile(TARGET_ZIP, "r") as zf:
        members = zf.infolist()
        total = len(members)
        print(f"      Extracting {total} files...")
        zf.extractall(DEST_DIR)
    print(f"      Extraction finished in {time.time() - t0:.1f}s.")
    
    if not exe_path.exists():
        raise FileNotFoundError(f"Extraction failed: {exe_path} not found!")
    print(f"      SUCCESS: CarlaUE4.exe verified at {exe_path}")


def main():
    print("=" * 70)
    print("TRAJPILOT CITYFLOW - AUTOMATED CARLA 0.9.16 INSTALLATION")
    print("=" * 70)
    check_url()
    current_size = prepare_zip_file()
    download_remaining(current_size)
    verify_archive()
    extract_carla()
    print("=" * 70)
    print("CARLA 0.9.16 READY AT C:\\CARLA_0.9.16")
    print("=" * 70)


if __name__ == "__main__":
    main()
