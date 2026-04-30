import json
import locale
import os


def load_language_list(language: str) -> dict[str, str]:
    with open(f"./i18n/locale/{language}.json", "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    language: str
    language_map: dict[str, str]

    def __init__(self, language: str | None = None) -> None:
        if language in ["Auto", None]:
            language = locale.getlocale()[0] or "en_US"
        if not os.path.exists(f"./i18n/locale/{language}.json"):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key: str) -> str:
        return self.language_map.get(key, key)

    def __repr__(self) -> str:
        return f"Use Language: {self.language}"
