import argparse
from email.parser import Parser
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


parser = argparse.ArgumentParser(description="Run stable ABI tests against an existing klib wheel")
parser.add_argument("wheel", type=Path)
args = parser.parse_args()
wheel = args.wheel.resolve()
with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory(prefix="klib-wheel-") as directory:
    names = archive.namelist()
    metadata = Parser().parsestr(archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode())
    assert metadata["Requires-Python"] == ">=3.12"
    assert "torch>=2.13" in metadata.get_all("Requires-Dist", [])
    assert not any("backend" in dependency for dependency in metadata.get_all("Requires-Dist", []))
    extension = next(n for n in names if n.endswith("/_stable.abi3.so"))
    digest = hashlib.sha256(archive.read(extension)).hexdigest()
    archive.extractall(directory)
    env = dict(os.environ, PYTHONPATH=directory, KLIB_EXPECTED_EXTENSION_SHA256=digest, OMP_NUM_THREADS="1")
    print(f"Testing {wheel.name}: SHA256 {hashlib.sha256(wheel.read_bytes()).hexdigest()}", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("test_stable.py").resolve())],
                   cwd=directory, env=env, check=True)
