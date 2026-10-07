#!/usr/bin/env python3
import csv, glob, json, os, re
from collections import defaultdict

OUT = os.environ.get("LAB_OUT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
FINAL = os.path.join(OUT, "final")
os.makedirs(FINAL, exist_ok=True)
OPS = ["OP1", "OP2", "OP3", "OP4", "OP5"]

def bul(o, adlar):
    if isinstance(o, dict):
        for a in adlar:
            if o.get(a) not in (None, ""):
                return o[a]
        for v in o.values():
            r = bul(v, adlar)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = bul(v, adlar)
            if r is not None:
                return r
    return None

def fb_mi(rec):
    gm = rec.get("generation_metadata")
    v = gm.get("fallback") if isinstance(gm, dict) else bul(rec, ["fallback"])
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "0", "none")
    return bool(v)

def evet(x):
    return str(x).strip().lower() in ("true", "1")

satirlar = []
ozet = defaultdict(lambda: defaultdict(int))
oran = defaultdict(list)
runlar = sorted([d for d in glob.glob(OUT + "/run_*") if re.search(r"/run_(\d+)$", d)],
                key=lambda d: int(re.search(r"run_(\d+)$", d).group(1)))
for d in runlar:
    n = int(re.search(r"run_(\d+)$", d).group(1))
    csvs = sorted(glob.glob(d + "/executed_testcases_*.csv"))
    if not csvs:
        print("run", n, "icin sonuc CSV'si yok, atlandi")
        continue
    fb = set()
    for f in glob.glob(d + "/.checkpoints/*/execution.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and fb_mi(rec):
                fb.add((str(bul(rec, ["generator"])), str(bul(rec, ["prompt_variant", "variant"])),
                        str(bul(rec, ["operation_id", "op_id"])), str(bul(rec, ["tc_id"]))))
    rows = list(csv.DictReader(open(csvs[-1], encoding="utf-8-sig")))
    hucre = defaultdict(int)
    sayac = defaultdict(int)
    gp = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        g, v, op, tc = r["generator"], r["prompt_variant"], r["operation_id"], r["tc_id"]
        hucre[(g, v, op)] += 1
        if not (r.get("actual_status") or "").strip():
            durum = "bos_istek"
        elif (g, v, op, tc) in fb:
            durum = "fallback"
        else:
            durum = "saglam"
        sayac[durum] += 1
        ozet[g][durum] += 1
        ozet[g]["uretilen"] += 1
        if durum == "saglam":
            p = 1 if evet(r.get("pass")) else 0
            gp[g][0] += 1
            gp[g][1] += p
            if op != "OP1":
                gp[g][2] += 1
                gp[g][3] += p
        satirlar.append([n, g, v, op, tc, r.get("pass"), r.get("actual_status"), durum])
    eksik = 0
    for g in sorted({r["generator"] for r in rows}):
        trad = g.startswith("Traditional")
        vs = ["traditional"] if trad else ["basic", "edge_focused"]
        bek = 10 if trad else 5
        e = sum(max(0, bek - hucre[(g, v, op)]) for v in vs for op in OPS)
        ozet[g]["uretilemedi"] += e
        ozet[g]["planlanan"] += 50
        eksik += e
        a = gp[g]
        if a[0]:
            oran[(g, "tum")].append(100.0 * a[1] / a[0])
        if a[2]:
            oran[(g, "op1_haric")].append(100.0 * a[3] / a[2])
    print("run %d: CSV %d satir | saglam %d | fallback %d | bos_istek %d | uretilemedi %d"
          % (n, len(rows), sayac["saglam"], sayac["fallback"], sayac["bos_istek"], eksik))
    if len(fb) != sayac["fallback"]:
        print("  UYARI: execution.jsonl'da %d fallback isaretli, CSV ile eslesen %d (anahtar adlari uyusmuyor olabilir)"
              % (len(fb), sayac["fallback"]))

def ort(x):
    return round(sum(x) / len(x), 1) if x else ""

kolon = ["generator", "planlanan", "uretilen", "saglam", "fallback", "bos_istek", "uretilemedi",
         "saglam_yuzde", "pass_ort", "pass_min", "pass_max", "pass_op1_haric_ort"]
tablo = []
for g in sorted(ozet):
    o = ozet[g]
    pt = oran[(g, "tum")]
    tablo.append([g, o["planlanan"], o["uretilen"], o["saglam"], o["fallback"], o["bos_istek"],
                  o["uretilemedi"], round(100.0 * o["saglam"] / o["planlanan"], 1) if o["planlanan"] else "",
                  ort(pt), round(min(pt), 1) if pt else "", round(max(pt), 1) if pt else "",
                  ort(oran[(g, "op1_haric")])])
with open(FINAL + "/siniflandirma.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "generator", "prompt_variant", "operation_id", "tc_id", "pass", "actual_status", "satir_durumu"])
    w.writerows(satirlar)
with open(FINAL + "/ozet_durum.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(kolon)
    w.writerows(tablo)
print()
print("\t".join(kolon))
for t in tablo:
    print("\t".join(str(x) for x in t))
print("\nYazildi: " + FINAL + "/siniflandirma.csv ve ozet_durum.csv")
