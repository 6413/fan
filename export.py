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

script_dir = os.path.dirname(os.path.abspath(__file__))

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
        try:
            found.add(s.decode())
        except:
            pass

for path in found:
    src = os.path.join(script_dir, path)
    dst = os.path.join(out_dir, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"copied: {path}")
    else:
        print(f"MISSING: {path}")

os.makedirs(out_dir, exist_ok=True)
exe_dst = os.path.join(out_dir, os.path.basename(exe))
shutil.copy2(exe, exe_dst)
print(f"copied exe: {exe_dst}")

imgui_src = os.path.join(script_dir, "imgui.ini")
if os.path.exists(imgui_src):
    shutil.copy2(imgui_src, os.path.join(out_dir, "imgui.ini"))
    print("copied: imgui.ini")