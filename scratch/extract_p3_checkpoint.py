import os
import sys
import time
import urllib.request
import tarfile

TARGET_DIR = r"D:\NLP_Project\scratch\checkpoints"
os.makedirs(TARGET_DIR, exist_ok=True)

url = "https://www.kaggle.com/api/v1/models/pengchengshi/dataset660_26classes_resize224_4661/pytorch/default/2/download"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

print(f"Connecting to Kaggle official model endpoint: {url}...")
t_start = time.time()
resp = urllib.request.urlopen(req, timeout=60)
print(f"Connected. Content-Length: {resp.headers.get('Content-Length')}")

# Wrapped response to log progress
class ProgressReader:
    def __init__(self, f):
        self.f = f
        self.total_read = 0
        self.last_log = time.time()
        self.last_bytes = 0

    def read(self, size=-1):
        data = self.f.read(size)
        self.total_read += len(data)
        now = time.time()
        if now - self.last_log >= 10.0:
            speed = (self.total_read - self.last_bytes) / (now - self.last_log) / (1024 * 1024)
            print(f"[{now - t_start:.1f}s] Downloaded {self.total_read / (1024 * 1024):.2f} MB ({speed:.2f} MB/s)")
            sys.stdout.flush()
            self.last_log = now
            self.last_bytes = self.total_read
        return data

reader = ProgressReader(resp)

print("Streaming and extracting fold_0/checkpoint_final.pth...")
sys.stdout.flush()

try:
    with tarfile.open(mode="r|gz", fileobj=reader) as tar:
        for member in tar:
            print(f"Found archive member: {member.name} ({member.size / (1024*1024):.2f} MB)")
            sys.stdout.flush()
            if member.name.endswith(".json") or member.name.endswith("fold_0/checkpoint_final.pth"):
                print(f"Extracting {member.name}...")
                sys.stdout.flush()
                tar.extract(member, path=TARGET_DIR)
                print(f"Successfully extracted {member.name}!")
                sys.stdout.flush()
                if member.name.endswith("fold_0/checkpoint_final.pth"):
                    print("Primary P3 Stage-2 checkpoint (fold 0) extraction COMPLETE!")
                    break
except Exception as e:
    print(f"Extraction finished or interrupted: {e}")
    sys.stdout.flush()

total_dur = time.time() - t_start
print(f"Done in {total_dur:.1f}s. Target dir contents:")
for root, dirs, files in os.walk(TARGET_DIR):
    for f in files:
        fp = os.path.join(root, f)
        print(f" - {fp} ({os.path.getsize(fp)/(1024*1024):.2f} MB)")
