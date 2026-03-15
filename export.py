import re, sys, os, shutil

def scan_binary(path):
  with open(path, "rb") as f:
    data = f.read()
  return re.findall(rb'[\x20-\x7e]{4,}', data)

exe = sys.argv[1]
out_dir = sys.argv[2]
force = len(sys.argv) > 3 and sys.argv[3] == '--force'

extensions = (
  b'.webp', b'.png', b'.jpg', b'.jpeg',
  b'.json', b'.sac', b'.wav', b'.ogg', b'.mp3',
  b'.glsl', b'.vert', b'.frag', b'.vs', b'.fs',
  b'.ttf', b'.otf', b'.woff', b'.woff2'
)

if os.path.exists(out_dir) and os.listdir(out_dir):
  if force or input(f"'{out_dir}' is not empty. Clear it? [y/N]: ").strip().lower() == 'y':
    try:
      shutil.rmtree(out_dir)
    except PermissionError:
      pass
    print(f"cleared: {out_dir}")

found = set()
for s in scan_binary(exe):
  if any(s.endswith(e) for e in extensions):
    found.add(s.decode())

for path in found:
  dst = os.path.join(out_dir, path)
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  if os.path.exists(path):
    shutil.copy2(path, dst)
    print(f"copied: {path}")
  else:
    print(f"MISSING: {path}")

exe_dst = os.path.join(out_dir, os.path.basename(exe))
os.makedirs(out_dir, exist_ok=True)
shutil.copy2(exe, exe_dst)
print(f"copied exe: {exe_dst}")