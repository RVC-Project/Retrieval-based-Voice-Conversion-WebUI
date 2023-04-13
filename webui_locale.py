import locale
import json

def load_language_list(language):
    with open(f"./locale/{language}.json", "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list

class I18nAuto:
    def __init__(self, language=None):
        if language is None:
            language = 'auto'
        if language == 'auto':
            language = locale.getdefaultlocale()[0]
        self.language = language
        print("Use Language:", language)
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map[key]
