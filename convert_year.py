import re
from functools import reduce


# ──────────────────────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────────────────────
MINGUO_OFFSET = 1911          # 民國 → 西元 的換算基準
MAX_HORIZON = 30              # 相對時程允許的最大年數
MAX_NUMERIC_ITEMS = 20        # 量化數據計數上限
YEAR_FLOOR, YEAR_CEIL = 2000, 2049   # 合法目標年份範圍


# ──────────────────────────────────────────────────────────────
# 中文數字解析
# ──────────────────────────────────────────────────────────────
_CHINESE_DIGIT = {
    "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def parse_chinese_number(token):
    """把中文或阿拉伯數字串轉成整數；無法解析回傳 None。"""
    if token.isdigit():
        return int(token)

    if token in _CHINESE_DIGIT:
        return _CHINESE_DIGIT[token]

    # 處理「十X」與「X十」兩位數型態
    if len(token) == 2:
        head, tail = token[0], token[1]
        if head == "十":
            return 10 + _CHINESE_DIGIT.get(tail, 0)
        if tail == "十":
            return _CHINESE_DIGIT.get(head, 0) * 10

    return None


# ──────────────────────────────────────────────────────────────
# 正則樣式集中管理
# ──────────────────────────────────────────────────────────────
class Patterns:
    western_year = re.compile(r"20(?:[0-4][0-9])")
    minguo_year = re.compile(r"民國\s*(\d{2,3})")

    _qty = r"(\d{1,2}|[一二兩三四五六七八九十]{1,2})"
    forward_phrases = [
        re.compile(r"未來\s*" + _qty + r"\s*年"),
        re.compile(r"今後\s*" + _qty + r"\s*年"),
        re.compile(_qty + r"\s*年(?:之?內|以內)"),
        re.compile(_qty + r"\s*年(?:之?後|以後)"),
    ]

    retrospective = re.compile(r"(?:過去|近|前|已)\s*\d{0,2}\s*年|年來")
    numeric_literal = re.compile(r"\d+(?:\.\d+)?")


# ──────────────────────────────────────────────────────────────
# 相對時程抽取
# ──────────────────────────────────────────────────────────────
def extract_relative_horizon(text):
    """
    從文字抽取最大的「未來型」相對年數。
    若某匹配的前文或本身帶有回顧語意則略過。
    """
    content = str(text)
    candidates = []

    for pattern in Patterns.forward_phrases:
        for hit in pattern.finditer(content):
            lookbehind = content[max(0, hit.start() - 3): hit.start()]
            if Patterns.retrospective.search(lookbehind + hit.group(0)):
                continue

            value = parse_chinese_number(hit.group(1))
            if value is not None and 1 <= value <= MAX_HORIZON:
                candidates.append(value)

    return max(candidates) if candidates else None


# ──────────────────────────────────────────────────────────────
# 年份相關推算
# ──────────────────────────────────────────────────────────────
def infer_report_year(url, fallback=2024):
    """報告年份取 URL 中最後出現的西元年；無則用 fallback。"""
    matches = Patterns.western_year.findall(str(url))
    return int(matches[-1]) if matches else fallback


def collect_target_years(text):
    """蒐集文字中所有落在合法區間的目標年份（含民國換算）。"""
    content = str(text)

    years = [int(y) for y in Patterns.western_year.findall(content)]
    years.extend(
        int(m) + MINGUO_OFFSET
        for m in Patterns.minguo_year.findall(content)
        if 100 <= int(m) <= 160
    )

    return [y for y in years if YEAR_FLOOR <= y <= YEAR_CEIL]


def absolute_horizon(text, report_yr):
    """以絕對年份計算距報告年份的最遠未來差距。"""
    future = [y for y in collect_target_years(text) if y >= report_yr]
    return (max(future) - report_yr) if future else None


# 與原始 future_gap 完全等價：僅取絕對年份差距
def horizon_gap(text, report_yr):
    return absolute_horizon(text, report_yr)


# ──────────────────────────────────────────────────────────────
# 特徵前綴
# ──────────────────────────────────────────────────────────────
def horizon_prefix(gap):
    """依差距產生時程前綴字串；NaN / None 視為未提及。"""
    if gap is None or gap != gap:          # gap != gap 偵測 NaN
        return "（未提及目標年份）"

    gap = int(gap)
    if gap == 0:
        return "（目標為當年）"
    return f"（目標距今{gap}年）"


def count_numeric_items(text):
    """計算量化數據項數，封頂於 MAX_NUMERIC_ITEMS。"""
    found = len(Patterns.numeric_literal.findall(str(text)))
    return found if found <= MAX_NUMERIC_ITEMS else MAX_NUMERIC_ITEMS


def numeric_prefix(text):
    return f"（量化數據{count_numeric_items(text)}項）"


def attach_time_prefix(text, url):
    """僅注入時程前綴（等價於原 inject）。"""
    return horizon_prefix(horizon_gap(str(text), infer_report_year(url))) + str(text)


def attach_features(text, url, use_time=True, use_numeric=False):
    """
    依旗標組合前綴並注入文字。
    順序與原 inject_features 一致：先數量、後時程。
    """
    segments = []
    if use_numeric:
        segments.append(numeric_prefix(text))
    if use_time:
        segments.append(horizon_prefix(horizon_gap(text, infer_report_year(url))))

    return reduce(lambda acc, seg: acc + seg, segments, "") + str(text)


# ──────────────────────────────────────────────────────────────
# 向後相容別名（保留原始公開介面名稱）
# ──────────────────────────────────────────────────────────────
relative_gap = extract_relative_horizon
report_year = infer_report_year
target_years = collect_target_years
future_gap_abs = absolute_horizon
future_gap = horizon_gap
gap_prefix = horizon_prefix
inject = attach_time_prefix
num_count = count_numeric_items
conc_prefix = numeric_prefix
inject_features = attach_features


# ──────────────────────────────────────────────────────────────
# 分析腳本
# ──────────────────────────────────────────────────────────────
def _run_analysis():
    import os
    import sys
    import pandas as pd

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    train = pd.read_csv(f"{base_dir}/vpesg_4k_train_1000.csv", encoding="utf-8-sig")
    valid = pd.read_csv(f"{base_dir}/vpesg4k_val_1000.csv", encoding="utf-8-sig")
    test = pd.read_csv(f"{base_dir}/vpesg4k_test_2000.csv", encoding="utf-8-sig")

    frame = pd.concat([train, valid], ignore_index=True)
    frame["verification_timeline"] = frame["verification_timeline"].fillna("N/A")
    frame["ry"] = frame["pdf_url"].map(infer_report_year)
    frame["gap"] = frame.apply(lambda row: horizon_gap(row["data"], row["ry"]), axis=1)

    print("report_year (train+val):", frame["ry"].value_counts().to_dict())
    print("report_year (test)     :", test["pdf_url"].map(infer_report_year).value_counts().to_dict())

    test_gap = test.apply(
        lambda row: horizon_gap(row["data"], infer_report_year(row["pdf_url"])), axis=1
    )
    print(
        "gap coverage (any horizon found):",
        f"{frame['gap'].notna().mean():.1%} train+val, {test_gap.notna().mean():.1%} test",
    )

    frame["abs_only"] = frame.apply(
        lambda row: absolute_horizon(row["data"], row["ry"]), axis=1
    )
    frame["rel"] = frame["data"].map(extract_relative_horizon)

    only_relative = frame["abs_only"].isna() & frame["rel"].notna()
    print(
        f"absolute-only coverage: {frame['abs_only'].notna().mean():.1%}"
        f"  -> +relative adds {only_relative.mean():.1%} rows"
        f" (now {frame['gap'].notna().mean():.1%})"
    )

    promised_relative = frame[(frame.promise_status == "Yes") & only_relative]
    print(
        f"\nrelative-ONLY rows (promise=Yes, n={len(promised_relative)})"
        " -> timeline distribution:"
    )
    print(promised_relative["verification_timeline"].value_counts().to_dict())

    rel_buckets = pd.cut(
        promised_relative["rel"], [0, 2, 5, 99], labels=["1-2", "3-5", "6+"]
    )
    print("rel-horizon bucket vs timeline:")
    print(pd.crosstab(rel_buckets, promised_relative["verification_timeline"]))

    promised = frame[frame.promise_status == "Yes"]
    print("\ngap-bucket -> timeline (promise=Yes):")
    gap_buckets = pd.cut(
        promised["gap"], [-1, 0, 2, 5, 99], labels=["0", "1-2", "3-5", "6+"]
    )
    print(pd.crosstab(gap_buckets, promised["verification_timeline"]))

    print("\n20 sampled injections (ROC rows prioritised):")
    roc_rows = frame[frame["data"].str.contains("民國")].index[:6]
    sampled = list(roc_rows) + list(frame.sample(14, random_state=1).index)

    for idx in sampled[:20]:
        row = frame.loc[idx]
        print(
            f"  ry={row['ry']} gap={row['gap']}"
            f" tl={row['verification_timeline']:22s}"
            f" | {horizon_prefix(row['gap'])} {row['data'][:55]}"
        )


if __name__ == "__main__":
    _run_analysis()