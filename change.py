import os
import re

folder = "data/lotka"  # Adjust to your folder path

for filename in os.listdir(folder):
    match = re.search(r'repeat(\d+)', filename)
    if match:
        old_repeat = int(match.group(1))
        if 20 <= old_repeat <= 40:
            new_repeat = old_repeat - 20  # shift into 0–20 range
            new_filename = re.sub(r'repeat\d+', f'repeat{new_repeat}', filename)

            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)

            if os.path.exists(new_path):
                print(f"Skipped (already exists): {new_filename}")
            else:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")