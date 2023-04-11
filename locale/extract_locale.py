import json
import re

# Define regular expression patterns
pattern = r'i18n\("([^"]+)"\)'

# Load the infer-webui.py file
with open('infer-web.py', 'r', encoding='utf-8') as f:
    contents = f.read()

# Matching with regular expressions
matches = re.findall(pattern, contents)

# Convert to key/value pairs
data = {}
for match in matches:
    key = match.strip('()"')
    value = ''
    data[key] = value


# Save as a JSON file
with open('labels.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
