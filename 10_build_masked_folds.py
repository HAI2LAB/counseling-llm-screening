# -*- coding: utf-8 -*-
"""실험② 마스킹 (v2: '술' 경계 처리 — 기술/예술/수술/미술/마술/저술/출술 오마스킹 방지)"""
import json, glob, os, re, collections

SRC, DST = "data/folds", "data/folds_masked"
TERMS = ["우울증","불안장애","공황장애","공황","조울증","알코올 중독","알콜 중독",
         "게임 중독","도박 중독","중독","알코올","알콜","음주","술자리","도박",
         "정신건강의학과","정신과","심리상담센터","상담센터","중독관리통합지원센터",
         "선별검사","심리검사","척도검사","진단","처방","항우울제","항불안제","수면제",
         "약물치료","금단","단주","단도박","재발","PHQ","GAD","BDI","AUDIT"]
TOKEN = "[가림]"
base = re.compile("|".join(re.escape(t) for t in sorted(TERMS, key=len, reverse=True)))
# '술' 단독: 앞이 기/예/수/미/마/저/출/학/의 가 아니고, 뒤가 조사/동사 시작이면 마스킹
sul = re.compile(r"(?<![기예수미마저출학의])술(?=[을이가도만에은는로와과랑 \.,]|먹|마시|취|끊|드시|한잔|자리)")

cnt = collections.Counter(); nrec = 0
def mask(v):
    hits = base.findall(v)
    for h in hits: cnt[h] += 1
    v = base.sub(TOKEN, v)
    n = len(sul.findall(v)); cnt["술(경계)"] += n
    return sul.sub(TOKEN, v)

for src in sorted(glob.glob(f"{SRC}/fold*/*.jsonl")):
    dst = src.replace(SRC, DST)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line); nrec += 1
            for k, v in r.items():
                if isinstance(v, str) and len(v) > 30: r[k] = mask(v)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[완료] {nrec}레코드 → {DST}")
print("[치환 상위 15]", cnt.most_common(15))
open("data/masking_dict.txt","w",encoding="utf-8").write("\n".join(TERMS + ["술(경계 규칙)"]))
