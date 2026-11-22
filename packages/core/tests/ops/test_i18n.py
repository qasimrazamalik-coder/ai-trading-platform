import re

from quant_os_core.ops import i18n
from quant_os_core.ops.dashboard_page import PAGE

ARABIC_SCRIPT = re.compile(r"[؀-ۿ]")


def test_every_key_present_and_non_empty_in_all_languages() -> None:
    keys = set(i18n.EN)
    for lang in i18n.LANGUAGES:
        table = i18n.STRINGS[lang]
        assert set(table) == keys, lang
        assert all(value.strip() for value in table.values()), lang


def test_urdu_and_arabic_are_native_script_with_same_placeholders() -> None:
    for lang in ("ur", "ar"):
        for key, text in i18n.STRINGS[lang].items():
            english = i18n.EN[key]
            assert set(re.findall(r"{\w+}", text)) == set(re.findall(r"{\w+}", english)), key
            assert text.count("<code>") == english.count("<code>"), key
            if key.startswith("legend_sma"):
                continue  # indicator names stay "SMA 50"
            assert ARABIC_SCRIPT.search(text), (lang, key)


def test_bundle_direction_speech_and_fallback() -> None:
    assert i18n.bundle("ur")["dir"] == "rtl" and i18n.bundle("ur")["speech"] == "ur-PK"
    assert i18n.bundle("ar-SA")["lang"] == "ar"
    assert i18n.bundle("fr")["lang"] == "en" and i18n.bundle(None)["dir"] == "ltr"


def test_every_page_label_has_a_string() -> None:
    used = set(re.findall(r'data-i[hpa]?="(\w+)"', PAGE))
    used |= set(re.findall(r'\bt\("(\w+)"[,)]', PAGE))
    assert used and used <= set(i18n.EN), used - set(i18n.EN)
