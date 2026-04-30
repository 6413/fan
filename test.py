import os
import re

# Types that come from cstdint/cstddef/cstdlib that end up needing std:: prefix
# when using import std; instead of #include <cstdint> etc.
CSTD_TYPES = {
    # cstdint
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int_least8_t", "int_least16_t", "int_least32_t", "int_least64_t",
    "uint_least8_t", "uint_least16_t", "uint_least32_t", "uint_least64_t",
    "int_fast8_t", "int_fast16_t", "int_fast32_t", "int_fast64_t",
    "uint_fast8_t", "uint_fast16_t", "uint_fast32_t", "uint_fast64_t",
    "intptr_t", "uintptr_t", "intmax_t", "uintmax_t",
    # cstddef
    "size_t", "ptrdiff_t", "nullptr_t", "max_align_t", "byte",
    # cstdlib
    "div_t", "ldiv_t", "lldiv_t",
}

EXTERNAL_DIRS = {"imgui", "nativefiledialog", "stb"}

def is_fan_file(filepath):
    parts = set(filepath.replace("\\", "/").split("/"))
    return not parts & EXTERNAL_DIRS

# Regex: match a type name NOT already preceded by "std::" and NOT followed by "::"
# Also skip if inside a #include line or a comment
# We build one big alternation pattern
TYPE_PATTERN = re.compile(
    r'(?<!std::)(?<!::)\b(' + '|'.join(re.escape(t) for t in sorted(CSTD_TYPES, key=len, reverse=True)) + r')\b(?!::)'
)

def transform_line(line):
    stripped = line.strip()
    # skip preprocessor, comments, and string literals (best effort)
    if stripped.startswith('#') or stripped.startswith('//'):
        return line
    return TYPE_PATTERN.sub(r'std::\1', line)

def transform_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = [transform_line(l) for l in lines]

    if new_lines != lines:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"  transformed: {filepath}")

for root, _, files in os.walk("./fan"):
    for file in files:
        filepath = os.path.join(root, file)
        if not is_fan_file(filepath):
            continue
        if file.endswith(".ixx") or file.endswith(".cpp"):
            transform_file(filepath)

print("Done.")
