"""Bulgu 1: modeller JSON'un icine kod ifadesi yazinca tum case'ler kaybolmasin.

Gercek olay: Gemini ve Groq cikti icinde "a".repeat(2000) gibi JavaScript
ifadeleri uretti. Bu, diziyi gecersiz JSON yapiyor ve json.loads tek bir
noktada patlayinca 15 case'in TAMAMI kayboluyordu.

Kurtarma BILEREK yorumlamaz: bozuk ifade literal'e cevrilmez, yalnizca o nesne
dusurulur. Boylece "model gecersiz cikti uretti" olcumu korunur.
"""

import json

import pytest

from generators.base import (
    _salvage_json_objects,
    _split_top_level_objects,
    extract_json_array,
    last_salvage_stats,
)

GOOD = '{"tc_id": "EP1_TC%d", "title": "ok %d"}'
BROKEN = '{"tc_id": "EP1_TC9", "query_params": {"long": "a".repeat(2000)}}'


def _array(*chunks):
    return "[\n  " + ",\n  ".join(chunks) + "\n]"


# ── Gerçek dünyadan gelen desen ──────────────────────────────────────────

def test_repeat_expression_no_longer_destroys_the_whole_array():
    text = _array(*[GOOD % (i, i) for i in range(1, 15)], BROKEN)
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)          # tum dizi gercekten gecersiz

    items = extract_json_array(text)
    assert len(items) == 14, "bozuk olan DUSMELI, digerleri kurtulmali"
    assert all(item["tc_id"].startswith("EP1_TC") for item in items)
    assert last_salvage_stats() == {"recovered": 14, "discarded": 1}


def test_broken_object_is_not_interpreted():
    """'a'.repeat(2000) literal'e CEVRILMEZ; nesne atilir."""
    items = extract_json_array(_array(GOOD % (1, 1), BROKEN))
    assert len(items) == 1
    assert items[0]["tc_id"] == "EP1_TC1"
    assert not any("long" in item.get("query_params", {}) for item in items)


# NaN BILEREK listede yok: Python'un json modulu NaN'i KABUL eder (standart dısı
# ama varsayilan), dolayisiyla o dizi gecersiz olmaz ve kurtarma yoluna hic
# girilmez. Gercek olayda gorulen desenler asagidakiler.
@pytest.mark.parametrize("expression", [
    '"a".repeat(2000)', "'x'.repeat(10)", "new Array(100).join('a')", "undefined",
])
def test_various_code_expressions_only_drop_their_own_object(expression):
    bad = '{"tc_id": "BAD", "v": %s}' % expression
    items = extract_json_array(_array(GOOD % (1, 1), bad, GOOD % (2, 2)))
    assert [i["tc_id"] for i in items] == ["EP1_TC1", "EP1_TC2"]


# ── Geçerli JSON yolu DEĞİŞMEDİ ──────────────────────────────────────────

def test_valid_array_does_not_take_the_salvage_path():
    text = _array(GOOD % (1, 1), GOOD % (2, 2))
    items = extract_json_array(text)
    assert len(items) == 2
    # Gecerli JSON dogrudan json.loads ile cozulur; kurtarma istatistigi
    # bu cagridan ETKILENMEZ.
    assert json.loads(text) == items


def test_non_json_text_is_not_mined_for_objects():
    """Eski boru-isaretli bicim kendi ayristiricisina kalmali.

    Kurtarma yalnizca GERCEK bir [ ... ] araligi varsa devreye girer; aksi halde
    'LOGIN_TC1|Valid|POST /login|{"email":"a"}|200|OK' satirindan sahte bir case
    uretir ve dogru ayristiriciyi devre disi birakirdi.
    """
    legacy = 'LOGIN_TC1|Valid|POST /login|{"email":"a"}|200|OK'
    assert extract_json_array(legacy) == []


def test_empty_and_garbage_inputs():
    assert extract_json_array("") == []
    assert extract_json_array("tamamen alakasiz metin") == []
    assert extract_json_array("[]") == []
    assert extract_json_array(None) == []


# ── Parçalayıcı ───────────────────────────────────────────────────────────

def test_splitter_respects_strings_and_escapes():
    text = '[{"a": "}{ icinde susluler"}, {"b": "kacis \\" sonra }"}]'
    chunks = _split_top_level_objects(text)
    assert len(chunks) == 2, f"string icindeki susluler sayilmamali: {chunks}"


def test_splitter_handles_nested_objects():
    text = '[{"a": {"b": {"c": 1}}}, {"d": 2}]'
    chunks = _split_top_level_objects(text)
    assert len(chunks) == 2
    assert json.loads(chunks[0]) == {"a": {"b": {"c": 1}}}


def test_splitter_only_collects_objects_not_other_items():
    """Parcalayici yalnizca { ... } bloklarini toplar.

    Dizi icindeki sayi/metin/alt-dizi ogeleri hic parca olarak gorulmez; bu
    yuzden "atilan" sayilmazlar. 'discarded' yalnizca AYRISTIRILAMAYAN bir
    nesne blogu icin artar.
    """
    recovered = _salvage_json_objects('[{"a":1}, [1,2], "metin"]')
    assert recovered == [{"a": 1}]
    assert last_salvage_stats() == {"recovered": 1, "discarded": 0}

    recovered = _salvage_json_objects('[{"a":1}, {"b": undefined}]')
    assert recovered == [{"a": 1}]
    assert last_salvage_stats() == {"recovered": 1, "discarded": 1}


# ── Bugünkü gerçek yanıtlar üzerinde (defterden, API çağrısı yok) ─────────

def test_against_todays_real_broken_responses():
    """Bugunun defterindeki gercek bozuk yanitlar kurtarilabiliyor mu."""
    import glob

    paths = glob.glob("outputs/free_phase/.calls/*/calls.jsonl")
    if not paths:
        pytest.skip("defter yok (bu test yalnizca kosu sonrasi anlamli)")

    checked = 0
    for line in open(paths[0], encoding="utf-8"):
        record = json.loads(line)
        raw = record.get("raw_response") or ""
        if record.get("failed") or not raw.strip().startswith("["):
            continue
        try:
            json.loads(raw)
            continue          # zaten gecerli
        except json.JSONDecodeError:
            pass
        items = extract_json_array(raw)
        # Yanittaki TUM case'ler bozuksa (orn. tek case'lik repair yaniti,
        # icinde "A".repeat(10240) gibi kod ifadesi) bos sonuc dogrudur:
        # kurtarma onlari atar ve defter discarded_cases ile sayar.
        if not items and int(record.get("discarded_cases") or 0) > 0:
            checked += 1
            continue
        assert items, "bozuk yanittan HIC case kurtarilamadi"
        checked += 1
    if checked == 0:
        pytest.skip("defterde bozuk JSON yanit yok")
