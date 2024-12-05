from pathlib import Path
import os
import shutil

dirpath = Path.cwd() / "images"
backup_root = dirpath.parent / "backup"

for root, dirs, files in os.walk(dirpath, topdown=True):
    root_path = Path(root)
    # Comment this "if", if you want to create a backup of all the data
    if "saved_models" in root:
        dirs[:] = []  # Don't iterate into saved_models directories
        continue

    for file in files:
        relative_path = root_path.relative_to(dirpath)
        destination = backup_root / relative_path
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy(root_path / file, destination)
    
