import argparse, re, sys, os, shutil

def scan_binary(path):
    with open(path, "rb") as f:
        data = f.read()
    return re.findall(rb'[\x20-\x7e]{4,}', data)

parser = argparse.ArgumentParser(description="Copy executable assets into a distributable directory.")
parser.add_argument("exe", help="input executable")
parser.add_argument("out_dir", help="output directory")
parser.add_argument("--force", action="store_true", help="clear an existing output directory")
parser.add_argument(
    "-p", "--search-path", "--path",
    dest="search_paths",
    action="append",
    default=[],
    metavar="PATH",
    help="comma-separated asset search paths; may be provided multiple times",
)
args = parser.parse_args()

exe = args.exe
out_dir = args.out_dir
force = args.force

extensions = (
    b'.webp', b'.png', b'.jpg', b'.jpeg',
    b'.json', b'.sac', b'.wav', b'.ogg', b'.mp3',
    b'.glsl', b'.vert', b'.frag', b'.vs', b'.fs',
    b'.ttf', b'.otf', b'.woff', b'.woff2'
)

script_dir = os.path.dirname(os.path.abspath(__file__))

search_paths = []
requested_paths = [
    part.strip()
    for value in args.search_paths
    for part in value.split(",")
    if part.strip()
]
for path in [*requested_paths, script_dir]:
    path = os.path.abspath(os.path.expanduser(path))
    if path not in search_paths:
        search_paths.append(path)

if os.path.exists(out_dir) and os.listdir(out_dir):
    if force or input(f"'{out_dir}' is not empty. Clear it? [y/N]: ").strip().lower() == 'y':
        try:
            shutil.rmtree(out_dir)
        except OSError as e:
            print(f"ERROR: unable to clear '{out_dir}': {e}", file=sys.stderr)
            print("Close any running executable using that directory and try again.", file=sys.stderr)
            sys.exit(1)
        print(f"cleared: {out_dir}")

found = set()
for s in scan_binary(exe):
    if any(s.lower().endswith(e) for e in extensions):
        try:
            decoded = s.decode()
            decoded = decoded.replace("\\", "/")
            decoded = re.sub(r'^[^A-Za-z0-9_./-]+', '', decoded)
            if decoded.startswith("./"):
                decoded = decoded[2:]
            decoded = decoded.lstrip("/")
            if "/" not in decoded and decoded.startswith("."):
                continue
            found.add(decoded)
        except UnicodeDecodeError:
            pass

def find_source(path):
    for root in search_paths:
        candidate = os.path.join(root, *path.split("/"))
        if os.path.isfile(candidate):
            return candidate
    return None

for path in found:
    src = find_source(path)
    dst = os.path.join(out_dir, *path.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if src:
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
