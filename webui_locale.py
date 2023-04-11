import locale
import json

def load_language_list(language):
    with open(f"./locale/{language}.json", "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list

class I18nAuto:
    def __init__(self, language=None):
        self.language_map = {}
        if language is None:
            language = 'auto'
        if language == 'auto':
            language = locale.getdefaultlocale()[0]
        self.language = language
        self.language_list = load_language_list(language)
        self.read_language(self.language_list)

    def read_language(self, lang_dict: dict):
        self.language_map.update(lang_dict)

    def __call__(self, key):
        return self.language_map[key]
