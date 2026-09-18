# -*- coding: utf-8 -*-
"""
00_explore.py : 데이터 구조 탐색
- 폴더/파일명 패턴, 피험자 ID, 회기, 클래스 분포 확인
- 라벨링 JSON 스키마 샘플 출력 (이후 멀티태스크 설계에 사용)
사용: python 00_explore.py --root /home/jovyan/HJW_counseling
"""
import argparse, json, re, random
from pathlib import Path
from collections import Counter, defaultdict

CLASS_KEYWORDS = ["우울증", "불안장애", "중독", "일반군"]

def infer_class(path: Path):
    s = str(path)
    for c in CLASS_KEYWORDS:
        if c in s:
            return c
    return None

def infer_subject(fname: str):
    # 예: resource_depression_1_check_D081.txt -> D081
    m = re.search(r"_([A-Za-z]+\d+)\.(txt|json)$", fname)
    return m.group(1) if m else None

def infer_session(path: Path):
    m = re.search(r"(\d+)회기", str(path))
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)_check", path.name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/jovyan/HJW_counseling")
    args = ap.parse_args()
    root = Path(args.root)

    txts = sorted(root.rglob("*.txt"))
    jsons = sorted(root.rglob("*.json"))
    print(f"[파일 수] txt={len(txts)}  json={len(jsons)}")

    # 최상위 분할(Training/Validation 등) 확인
    tops = Counter(p.relative_to(root).parts[0] for p in txts)
    print(f"[최상위 폴더별 txt 수] {dict(tops)}")

    # 클래스 x 피험자 분포
    cls_cnt, subj_by_cls = Counter(), defaultdict(set)
    unparsed = []
    for p in txts:
        c, s = infer_class(p), infer_subject(p.name)
        if c is None or s is None:
            unparsed.append(str(p)); continue
        cls_cnt[c] += 1
        subj_by_cls[c].add(s)
    print(f"[클래스별 txt 수] {dict(cls_cnt)}")
    print(f"[클래스별 고유 피험자 수] " + str({k: len(v) for k, v in subj_by_cls.items()}))
    if unparsed:
        print(f"[경고] 클래스/피험자 파싱 실패 {len(unparsed)}건, 예시:")
        for u in unparsed[:5]:
            print("   ", u)

    # 피험자 ID 접두 문자 분포 (클래스별 ID 체계 확인)
    prefix = Counter()
    for subjs in subj_by_cls.values():
        for s in subjs:
            prefix[re.match(r"[A-Za-z]+", s).group(0)] += 1
    print(f"[피험자 ID 접두어 분포] {dict(prefix)}")

    # Training/Validation 간 피험자 중복 검사 (누수 확인 핵심)
    subj_by_top = defaultdict(set)
    for p in txts:
        s = infer_subject(p.name)
        if s:
            subj_by_top[p.relative_to(root).parts[0]].add(s)
    keys = list(subj_by_top.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ov = subj_by_top[keys[i]] & subj_by_top[keys[j]]
            print(f"[피험자 중복] {keys[i]} ∩ {keys[j]} = {len(ov)}명"
                  + (f" 예시 {sorted(ov)[:5]}" if ov else ""))

    # txt 길이 통계
    lens = []
    for p in random.sample(txts, min(200, len(txts))):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            t = ""
        lens.append(len(t))
    if lens:
        lens.sort()
        print(f"[txt 길이(문자), 표본 {len(lens)}] "
              f"min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")

    # txt 샘플 출력
    sample_txt = txts[0]
    print(f"\n===== TXT 샘플: {sample_txt} =====")
    print(sample_txt.read_text(encoding="utf-8", errors="ignore")[:1500])

    # JSON 스키마 샘플
    if jsons:
        sample_json = jsons[0]
        print(f"\n===== JSON 샘플: {sample_json} =====")
        try:
            obj = json.loads(sample_json.read_text(encoding="utf-8", errors="ignore"))
            def walk(o, depth=0, max_depth=3):
                pad = "  " * depth
                if depth > max_depth:
                    print(pad + "..."); return
                if isinstance(o, dict):
                    for k, v in list(o.items())[:15]:
                        print(f"{pad}{k}: ({type(v).__name__})")
                        walk(v, depth + 1, max_depth)
                elif isinstance(o, list):
                    print(f"{pad}[list len={len(o)}]")
                    if o:
                        walk(o[0], depth + 1, max_depth)
                else:
                    print(f"{pad}= {str(o)[:80]}")
            walk(obj)
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")

if __name__ == "__main__":
    main()
