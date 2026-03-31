import cv2
import os
import re

# ── config ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = "./generated"        # folder containing your epoch images
OUTPUT     = "training.mp4"
FPS        = 10                 # frames per second (increase for faster playback)
# ──────────────────────────────────────────────────────────────────────────────

def extract_epoch(filename):
    """Pull the epoch number out of the filename for correct sorting."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# Collect and sort images by epoch number
images = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
    key=extract_epoch
)

if not images:
    raise FileNotFoundError(f"No images found in '{IMAGE_DIR}'")

print(f"Found {len(images)} images — {images[0]} → {images[-1]}")

# Read first frame to get dimensions
first = cv2.imread(os.path.join(IMAGE_DIR, images[0]))
h, w  = first.shape[:2]

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    (w, h)
)

for fname in images:
    frame = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if frame is None:
        print(f"  skipping unreadable file: {fname}")
        continue
    writer.write(frame)

writer.release()
print(f"Saved → {OUTPUT}  ({len(images)} frames @ {FPS} fps)")