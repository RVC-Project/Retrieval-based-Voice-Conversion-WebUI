import json
import re

# Define regular expression patterns
pattern = r'i18n\([^)]*\)'

# Initialize the dictionary to store key-value pairs
data = {}

# Extract labels from infer-webui.py
with open('infer-web.py', 'r', encoding='utf-8') as f:
    contents = f.read()
    matches = re.findall(pattern, contents)
    for match in matches:
        key = match.strip('i18n()"\'')
        data[key] = key

# Extract labels from gui.py
with open('gui.py', 'r', encoding='utf-8') as f:
    contents = f.read()
    matches = re.findall(pattern, contents)
    for match in matches:
        key = match.strip('i18n()"\'')
        data[key] = key

# Save as a JSON file
with open('./locale/zh_CN.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
