import json
import re

# Define regular expression patterns
pattern = r"""i18n\([\s\n\t]*(["'][^"']+["'])[\s\n\t]*\)"""

# Initialize the dictionary to store key-value pairs
data = {}


def process(fn: str):
    global data
    with open(fn, "r", encoding="utf-8") as f:
        contents = f.read()
        matches = re.findall(pattern, contents)
        for key in matches:
            key = eval(key)
            print("extract:", key)
            data[key] = key


print("processing infer-web.py")
process("infer-web.py")

print("processing gui_v0.py")
process("gui_v0.py")

print("processing gui_v1.py")
process("gui_v1.py")

# Save as a JSON file
with open("./lib/i18n/zh_CN.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    f.write("\n")
