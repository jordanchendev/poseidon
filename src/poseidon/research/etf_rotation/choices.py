from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from poseidon.research.etf_rotation.pair_config import PAIR_ORDER, PAIRS

ROOT = Path("outputs/rotation-research-20260612")
RESULTS = ROOT / "results"
OUT = ROOT / "representative-choices-zhtw.md"


CHOICE_ZH = {
    "core_buy_hold": "純核心持有",
    "leveraged_buy_hold": "純槓桿持有",
    "current_mix_buy_hold": "目前比例持有",
    "best_return_under_35dd": "回撤小於 35% 內最高報酬",
    "best_return_under_40dd": "回撤小於 40% 內最高報酬",
    "best_return_under_45dd": "回撤小於 45% 內最高報酬",
    "best_return_under_50dd": "回撤小於 50% 內最高報酬",
    "best_return_under_60dd": "回撤小於 60% 內最高報酬",
    "best_score": "綜合分數最佳",
    "best_full_return": "歷史報酬最高",
    "best_1_stage_score": "單段規則最佳",
    "best_2_stage_score": "兩段規則最佳",
    "best_3_stage_score": "三段規則最佳",
    "best_low_turnover_return": "低交易次數最高報酬",
    "best_mid_turnover_return": "中交易次數最高報酬",
    "best_high_turnover_return": "高交易次數最高報酬",
    "best_de_risk_score": "反向降槓桿綜合最佳",
    "best_de_risk_return": "反向降槓桿歷史報酬最高",
    "best_de_risk_under_40dd": "反向降槓桿回撤小於 40% 內最高報酬",
    "best_de_risk_under_45dd": "反向降槓桿回撤小於 45% 內最高報酬",
    "best_de_risk_under_50dd": "反向降槓桿回撤小於 50% 內最高報酬",
    "best_de_risk_under_60dd": "反向降槓桿回撤小於 60% 內最高報酬",
    "best_de_risk_low_turnover_return": "反向降槓桿低交易次數最高報酬",
    "best_de_risk_2_stage_score": "反向降槓桿兩段規則最佳",
    "best_de_risk_3_stage_score": "反向降槓桿三段規則最佳",
    "best_de_risk_cash_score": "反向切現金綜合最佳",
    "best_de_risk_cash_return": "反向切現金歷史報酬最高",
    "best_de_risk_cash_under_40dd": "反向切現金回撤小於 40% 內最高報酬",
    "best_de_risk_cash_under_45dd": "反向切現金回撤小於 45% 內最高報酬",
    "best_de_risk_cash_under_50dd": "反向切現金回撤小於 50% 內最高報酬",
    "best_de_risk_cash_under_60dd": "反向切現金回撤小於 60% 內最高報酬",
    "best_de_risk_cash_low_turnover_return": "反向切現金低交易次數最高報酬",
    "best_de_risk_cash_2_stage_score": "反向切現金兩段規則最佳",
    "best_de_risk_cash_3_stage_score": "反向切現金三段規則最佳",
    "best_leveraged_cash_band_score": "正2現金權重帶綜合最佳",
    "best_leveraged_cash_band_return": "正2現金權重帶歷史報酬最高",
    "best_leveraged_cash_band_under_40dd": "正2現金權重帶回撤小於 40% 內最高報酬",
    "best_leveraged_cash_band_under_50dd": "正2現金權重帶回撤小於 50% 內最高報酬",
    "best_leveraged_cash_band_under_60dd": "正2現金權重帶回撤小於 60% 內最高報酬",
    "ma_sma200": "MA：核心站上 SMA200",
    "ma_ema200": "MA：核心站上 EMA200",
    "ma_sma200_band_2": "MA：SMA200 ±2% 權重帶",
}


DESCRIPTIONS = {
    "core_buy_hold": "完全不切換槓桿標的，只持有核心 ETF，適合拿來當最低風險基準。",
    "leveraged_buy_hold": "完全持有槓桿 ETF，報酬潛力最高但回撤也最極端。",
    "current_mix_buy_hold": "沿用你目前核心加少量槓桿的比例，不做任何再平衡切換。",
    "best_return_under_35dd": "在歷史最大回撤控制在約 35% 以內時，找最高終值的策略。",
    "best_return_under_40dd": "在歷史最大回撤控制在約 40% 以內時，找最高終值的策略。",
    "best_return_under_45dd": "在歷史最大回撤控制在約 45% 以內時，找最高終值的策略。",
    "best_return_under_50dd": "在歷史最大回撤控制在約 50% 以內時，找最高終值的策略。",
    "best_return_under_60dd": "在歷史最大回撤控制在約 60% 以內時，找最高終值的策略。",
    "best_score": "用報酬、Sharpe、回撤、潰瘍指數和交易次數一起評分後的最佳折衷策略。",
    "best_full_return": "完全以歷史終值最大化為目標，不優先限制回撤。",
    "best_1_stage_score": "只用一個進出門檻，在同類規則中找綜合分數最佳。",
    "best_2_stage_score": "用兩段分批切換槓桿，在同類規則中找綜合分數最佳。",
    "best_3_stage_score": "用三段更細分的切換規則，在同類規則中找綜合分數最佳。",
    "best_low_turnover_return": "限制歷史轉換次數不超過 10 次，在低操作頻率中找最高報酬。",
    "best_mid_turnover_return": "限制歷史轉換次數約 11 到 25 次，在中等操作頻率中找最高報酬。",
    "best_high_turnover_return": "允許超過 25 次轉換，在高操作頻率中找最高報酬。",
    "best_de_risk_score": "預設持有槓桿 ETF，遇到下跌時降到核心 ETF，用綜合分數找最佳折衷。",
    "best_de_risk_return": "預設持有槓桿 ETF，遇到下跌時降到核心 ETF，完全以歷史終值最大化為目標。",
    "best_de_risk_under_40dd": "預設持有槓桿 ETF，遇到下跌時降到核心 ETF，並把歷史最大回撤控制在約 40% 以內。",
    "best_de_risk_under_45dd": "預設持有槓桿 ETF，遇到下跌時降到核心 ETF，並把歷史最大回撤控制在約 45% 以內。",
    "best_de_risk_under_50dd": "預設持有槓桿 ETF，遇到下跌時降到核心 ETF，並把歷史最大回撤控制在約 50% 以內。",
    "best_de_risk_under_60dd": "預設持有槓桿 ETF，遇到下跌時降到核心 ETF，並把歷史最大回撤控制在約 60% 以內。",
    "best_de_risk_low_turnover_return": "預設持有槓桿 ETF，遇到下跌時降到核心 ETF，並限制歷史轉換次數不超過 10 次。",
    "best_de_risk_2_stage_score": "預設持有槓桿 ETF，遇到下跌時用兩段規則分批降槓桿，在同類規則中找綜合分數最佳。",
    "best_de_risk_3_stage_score": "預設持有槓桿 ETF，遇到下跌時用三段規則分批降槓桿，在同類規則中找綜合分數最佳。",
    "best_de_risk_cash_score": "預設持有槓桿 ETF，遇到下跌時切到現金，用綜合分數找最佳折衷。",
    "best_de_risk_cash_return": "預設持有槓桿 ETF，遇到下跌時切到現金，完全以歷史終值最大化為目標。",
    "best_de_risk_cash_under_40dd": "預設持有槓桿 ETF，遇到下跌時切到現金，並把歷史最大回撤控制在約 40% 以內。",
    "best_de_risk_cash_under_45dd": "預設持有槓桿 ETF，遇到下跌時切到現金，並把歷史最大回撤控制在約 45% 以內。",
    "best_de_risk_cash_under_50dd": "預設持有槓桿 ETF，遇到下跌時切到現金，並把歷史最大回撤控制在約 50% 以內。",
    "best_de_risk_cash_under_60dd": "預設持有槓桿 ETF，遇到下跌時切到現金，並把歷史最大回撤控制在約 60% 以內。",
    "best_de_risk_cash_low_turnover_return": "預設持有槓桿 ETF，遇到下跌時切到現金，並限制歷史轉換次數不超過 10 次。",
    "best_de_risk_cash_2_stage_score": "預設持有槓桿 ETF，遇到下跌時用兩段規則分批切現金，在同類規則中找綜合分數最佳。",
    "best_de_risk_cash_3_stage_score": "預設持有槓桿 ETF，遇到下跌時用三段規則分批切現金，在同類規則中找綜合分數最佳。",
    "best_leveraged_cash_band_score": "固定持有槓桿 ETF 加現金，當槓桿 ETF 權重偏離目標權重帶時再平衡。",
    "best_leveraged_cash_band_return": "固定持有槓桿 ETF 加現金，用權重帶再平衡，在同類規則中找最高終值。",
    "best_leveraged_cash_band_under_40dd": "固定持有槓桿 ETF 加現金，用權重帶再平衡，並把歷史最大回撤控制在約 40% 以內。",
    "best_leveraged_cash_band_under_50dd": "固定持有槓桿 ETF 加現金，用權重帶再平衡，並把歷史最大回撤控制在約 50% 以內。",
    "best_leveraged_cash_band_under_60dd": "固定持有槓桿 ETF 加現金，用權重帶再平衡，並把歷史最大回撤控制在約 60% 以內。",
    "ma_sma200": "趨勢濾網策略：核心 ETF 收盤站上 SMA200 才持有槓桿 ETF，跌回均線下方則回核心 ETF。",
    "ma_ema200": "趨勢濾網策略：核心 ETF 收盤站上 EMA200 才持有槓桿 ETF，EMA 對近期價格反應比 SMA 快。",
    "ma_sma200_band_2": "趨勢濾網策略：用 SMA200 上下 2% 權重帶降低均線附近來回洗出洗進。",
}


PAIR_ZH = {key: cfg.zh_name for key, cfg in PAIRS.items()}

PAIR_ASSETS = {
    key: {
        "core": cfg.core,
        "lev": cfg.lev,
        "current_core": cfg.current_core_twd,
        "current_lev": cfg.current_lev_twd,
    }
    for key, cfg in PAIRS.items()
}


def money(value: float) -> str:
    return f"{value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def rule(row: pd.Series) -> str:
    # Kept for backward compatibility; rule_text() is the human-readable version.
    if int(row["stages"]) == 0:
        return "不切換"
    return f"進場 {row['enter']}；出場 {row['exit']}；槓桿比例 {row['levels']}；floor {row['floor']}"


def percent(value: float) -> str:
    return f"{value * 100:g}%"


def split_values(text: str) -> list[float]:
    if text is None or pd.isna(text):
        return []
    return [float(part) for part in str(text).split("|") if part]


def base_holding_text(pair: str, floor: float) -> str:
    assets = PAIR_ASSETS[pair]
    if floor <= 0:
        return f"平常持有 100% {assets['core']}"
    if floor >= 1:
        return f"平常持有 100% {assets['lev']}"
    return f"平常持有 {percent(1 - floor)} {assets['core']} / {percent(floor)} {assets['lev']}"


def de_risk_base_holding_text(pair: str, max_lev: float) -> str:
    assets = PAIR_ASSETS[pair]
    if max_lev >= 1:
        return f"平常持有 100% {assets['lev']}"
    if max_lev <= 0:
        return f"平常持有 100% {assets['core']}"
    return f"平常持有 {percent(1 - max_lev)} {assets['core']} / {percent(max_lev)} {assets['lev']}"


def leveraged_cash_holding_text(pair: str, leveraged_weight: float) -> str:
    assets = PAIR_ASSETS[pair]
    if leveraged_weight >= 1:
        return f"平常持有 100% {assets['lev']}"
    if leveraged_weight <= 0:
        return "平常持有 100% 現金"
    return f"平常持有 {percent(leveraged_weight)} {assets['lev']} / {percent(1 - leveraged_weight)} 現金"


def current_mix_text(pair: str) -> str:
    assets = PAIR_ASSETS[pair]
    total = assets["current_core"] + assets["current_lev"]
    core_pct = assets["current_core"] / total
    lev_pct = assets["current_lev"] / total
    return f"不切換；長期持有目前比例：{percent(core_pct)} {assets['core']} / {percent(lev_pct)} {assets['lev']}"


def exit_rules(pair: str, exits: list[float], levels: list[float], floor: float) -> list[str]:
    assets = PAIR_ASSETS[pair]
    rules: list[str] = []
    if len(exits) == 1:
        rules.append(f"之後 {assets['core']} 從低點反彈 {percent(exits[0])} 時，降回 {percent(floor)} {assets['lev']}")
        return rules

    # The simulation reduces exposure gradually as rebound thresholds are reached:
    # first threshold -> previous/lower stage, final threshold -> floor.
    for index, threshold in enumerate(exits):
        target = floor if index == len(exits) - 1 else levels[len(exits) - 2 - index]
        rules.append(f"反彈 {percent(threshold)} 時，把 {assets['lev']} 降到 {percent(target)}")
    return rules


def de_risk_exit_rules(pair: str, exits: list[float], levels: list[float], max_lev: float) -> list[str]:
    assets = PAIR_ASSETS[pair]
    rules: list[str] = []
    if len(exits) == 1:
        rules.append(
            f"之後 {assets['core']} 從低點反彈 {percent(exits[0])} 時，回到 {percent(max_lev)} {assets['lev']}"
        )
        return rules

    # Reverse mode restores leveraged exposure gradually as rebound thresholds
    # are reached: first threshold -> previous/higher stage, final threshold -> max_lev.
    for index, threshold in enumerate(exits):
        target = max_lev if index == len(exits) - 1 else levels[len(exits) - 2 - index]
        rules.append(f"反彈 {percent(threshold)} 時，回升到 {percent(target)} {assets['lev']}")
    return rules


def de_risk_cash_exit_rules(pair: str, exits: list[float], levels: list[float], max_lev: float) -> list[str]:
    assets = PAIR_ASSETS[pair]
    rules: list[str] = []
    if len(exits) == 1:
        rules.append(
            f"之後 {assets['core']} 從低點反彈 {percent(exits[0])} 時，回到 {percent(max_lev)} {assets['lev']}"
        )
        return rules

    for index, threshold in enumerate(exits):
        target = max_lev if index == len(exits) - 1 else levels[len(exits) - 2 - index]
        rules.append(f"反彈 {percent(threshold)} 時，回升到 {percent(target)} {assets['lev']}，其餘持有現金")
    return rules


def rule_text(pair: str, row: pd.Series) -> str:
    assets = PAIR_ASSETS[pair]
    choice = row["choice"]
    if choice == "core_buy_hold":
        return f"不切換；全程持有 100% {assets['core']}。"
    if choice == "leveraged_buy_hold":
        return f"不切換；全程持有 100% {assets['lev']}。"
    if choice == "current_mix_buy_hold":
        return current_mix_text(pair) + "。"

    floor = float(row["floor"])
    max_lev = float(row["max_lev"])
    enters = split_values(row["enter"])
    exits = split_values(row["exit"])
    levels = split_values(row["levels"])
    mode = row.get("mode", "buy_dip")

    if mode == "de_risk_on_drawdown":
        parts = [de_risk_base_holding_text(pair, max_lev)]
        for threshold, level in zip(enters, levels, strict=True):
            parts.append(
                f"{assets['core']} 從近期高點下跌 {percent(threshold)} 時，降到 {percent(level)} {assets['lev']}"
            )
        parts.extend(de_risk_exit_rules(pair, exits, levels, max_lev))
        return "；".join(parts) + "。"

    if mode == "de_risk_to_cash_on_drawdown":
        parts = [leveraged_cash_holding_text(pair, max_lev)]
        for threshold, level in zip(enters, levels, strict=True):
            parts.append(
                f"{assets['core']} 從近期高點下跌 {percent(threshold)} 時，降到 {percent(level)} {assets['lev']}，其餘持有現金"
            )
        parts.extend(de_risk_cash_exit_rules(pair, exits, levels, max_lev))
        return "；".join(parts) + "。"

    if mode == "leveraged_cash_band":
        target = max_lev
        band = enters[0]
        lower = max(0.0, target - band)
        upper = min(1.0, target + band)
        return (
            f"{leveraged_cash_holding_text(pair, target)}；"
            f"{assets['lev']} 權重高於 {percent(upper)} 或低於 {percent(lower)} 時，再平衡回 {percent(target)} {assets['lev']}。"
        )

    if mode == "ma_sma":
        window = int(enters[0])
        return (
            f"{assets['core']} 收盤高於 SMA{window} 時，隔日持有 100% {assets['lev']}；否則持有 100% {assets['core']}。"
        )

    if mode == "ma_ema":
        window = int(enters[0])
        return (
            f"{assets['core']} 收盤高於 EMA{window} 時，隔日持有 100% {assets['lev']}；否則持有 100% {assets['core']}。"
        )

    if mode == "ma_sma_band":
        window = int(enters[0])
        band = enters[1]
        return (
            f"{assets['core']} 收盤高於 SMA{window} {percent(band)} 時，隔日持有 100% {assets['lev']}；"
            f"{assets['core']} 收盤低於 SMA{window} {percent(band)} 時，隔日回到 100% {assets['core']}。"
        )

    parts = [base_holding_text(pair, floor)]
    for threshold, level in zip(enters, levels, strict=True):
        parts.append(
            f"{assets['core']} 從近期高點下跌 {percent(threshold)} 時，把 {assets['lev']} 調到 {percent(level)}"
        )
    parts.extend(exit_rules(pair, exits, levels, floor))
    return "；".join(parts) + "。"


def cell(value: object) -> str:
    return str(value).replace("|", r"\|")


def projection(capital: float, cagr: float, years: int) -> float:
    return capital * ((1 + cagr) ** years)


def table_for_pair(
    df: pd.DataFrame, pair: str, capital_values: dict[str, float], capital_labels: dict[str, str]
) -> str:
    lines = [
        f"## {PAIR_ZH[pair]}",
        "",
        f"起始資金：`{capital_labels[pair]}`",
        "",
        "| 策略 | 一句話說明 | 規則 | CAGR | MaxDD | 轉換次數 | 1 年 | 3 年 | 5 年 | 7 年 | 10 年 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df[df["pair"] == pair].iterrows():
        choice = row["choice"]
        cagr = float(row["cagr"])
        capital = capital_values[pair]
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(CHOICE_ZH.get(choice, choice)),
                    cell(DESCRIPTIONS.get(choice, "")),
                    cell(rule_text(pair, row)),
                    pct(float(row["cagr"])),
                    pct(float(row["maxdd"])),
                    str(int(row["switches"])),
                    money(projection(capital, cagr, 1)),
                    money(projection(capital, cagr, 3)),
                    money(projection(capital, cagr, 5)),
                    money(projection(capital, cagr, 7)),
                    money(projection(capital, cagr, 10)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nasdaq-capital", type=float, default=PAIRS["NASDAQ"].capital_twd)
    parser.add_argument("--taiwan-capital", type=float, default=PAIRS["TAIWAN50"].capital_twd)
    parser.add_argument("--default-new-capital", type=float, default=1_000_000)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--title", default="代表性策略比較（繁中）")
    args = parser.parse_args()

    capital_values = {key: float(cfg.capital_twd) for key, cfg in PAIRS.items()}
    capital_values.update(
        {
            "NASDAQ": args.nasdaq_capital,
            "TAIWAN50": args.taiwan_capital,
        }
    )
    for key in PAIR_ORDER:
        if key not in {"NASDAQ", "TAIWAN50"}:
            capital_values[key] = args.default_new_capital
    capital_labels = {pair: f"TWD {capital_values[pair]:,.0f}" for pair in capital_values}

    df = pd.read_csv(RESULTS / "representative_choices.csv")
    content = [
        f"# {args.title}",
        "",
        "這份表不是只列最佳結果，而是把不同風險、交易頻率、分段方式的代表性策略全部列出來。",
        "",
        "年期預估是用各策略的歷史 CAGR 做幾何外推，目的是比較策略差距，不是保證未來報酬。",
        "",
        *(table_for_pair(df, pair, capital_values, capital_labels) for pair in PAIR_ORDER),
    ]
    out = Path(args.out)
    out.write_text("\n".join(content), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
