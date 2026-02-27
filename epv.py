# epv.py（x,y座標＋EPVヒートマップ描画付き）
# ================================================
# 変更点：
# ① 座標列「x座標」「y座標」を明示的に使用
# ② 保存後に Seaborn で EPV（points_scored）のヒートマップを描画
# ================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ========= ユーザー設定 =========
INPUT_CSV = "/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/possession_df_ver2.csv"
OUTPUT_FEATURES_CSV = "/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/epv_features_with_xy.csv"
TIME_COL_EXPLICIT = None  # 残時間列名がわかれば明示指定
# ===============================

def main():
    # ---------- 読み込み ----------
    x = pd.read_csv(INPUT_CSV)
    x = x.loc[:, ~x.columns.str.contains("^Unnamed")]
    x = x.sort_values(["試合ID", "ピリオド", "履歴No"]).reset_index(drop=True)

    # ---------- 基本チェック ----------
    need_cols = ["試合ID","ピリオド","履歴No","チームID","アクション1",
                 "possession_id","possession_team","x座標","y座標"]
    miss = [c for c in need_cols if c not in x.columns]
    if miss:
        raise ValueError(f"必要列が見つかりません: {miss}")

    # ---------- start/end フラグ ----------
    if "possession_start_flag" not in x.columns:
        print("⚠️ possession_start_flag が無いため自動生成します。")
        x["possession_start_flag"] = 0
        x.loc[x.groupby(["試合ID","possession_id"]).head(1).index, "possession_start_flag"] = 1
    if "possession_end_flag" not in x.columns:
        print("⚠️ possession_end_flag が無いため自動生成します。")
        x["possession_end_flag"] = 0
        x.loc[x.groupby(["試合ID","possession_id"]).tail(1).index, "possession_end_flag"] = 1

    # ---------- 得点列 ----------
    def calc_points(a1):
        if a1 in {1}: return 3
        if a1 in {3,4,44}: return 2
        if a1 in {7}: return 1
        return 0
    x["得点"] = x["アクション1"].map(calc_points).fillna(0).astype(int)

    # ---------- ポゼッション得点 ----------
    poss_points = (x.groupby(["試合ID","possession_id","possession_team"], as_index=False)
                     .agg(points_scored=("得点","sum")))

    # ---------- 累積スコア ----------
    x["team_score"] = (x.sort_values(["試合ID","チームID","ピリオド","履歴No"])
                         .groupby(["試合ID","チームID"])["得点"].cumsum())
    score_wide = (x.groupby(["試合ID","ピリオド","履歴No","チームID"])["team_score"]
                    .max().unstack(fill_value=0))
    score_wide.index = score_wide.index.set_names(["試合ID","ピリオド","履歴No"])

    # ---------- ポゼッション開始行 ----------
    starts = x[x["possession_start_flag"] == 1][[
        "試合ID","ピリオド","履歴No","チームID","possession_id","possession_team","x座標","y座標"
    ]].copy()

    # ---------- 点差算出 ----------
    starts_idx = starts.set_index(["試合ID","ピリオド","履歴No"])
    score_narrow = score_wide.groupby(level=[0,1,2]).max()
    starts_joined = starts_idx.join(score_narrow, how="left").reset_index()

    score_cols = [c for c in starts_joined.columns if str(c).isdigit()]

    def margin_from_row(row):
        tid = int(row["チームID"])
        vals = row[score_cols]
        my = row.get(tid, vals.get(str(tid), 0))
        opp_vals = [row[c] for c in score_cols if str(c) != str(tid)]
        opp = np.nanmax(opp_vals) if opp_vals else 0
        return float(my - opp)
    starts_joined["score_margin_start"] = starts_joined.apply(margin_from_row, axis=1)

    # ---------- 残り時間 ----------
    if TIME_COL_EXPLICIT and TIME_COL_EXPLICIT in x.columns:
        time_col = TIME_COL_EXPLICIT
    else:
        cand = [c for c in x.columns if "残" in c or "time" in c.lower()]
        time_col = cand[0] if cand else None
    if time_col:
        time_idx = x.set_index(["試合ID","ピリオド","履歴No"])[time_col]
        starts_joined["clock_start"] = time_idx.reindex(
            starts_joined.set_index(["試合ID","ピリオド","履歴No"]).index
        ).values
    else:
        starts_joined["clock_start"] = np.nan

    # ---------- 速攻・セカンドチャンス ----------
    def has_tag_any(g, tags):
        arrs = []
        for col in ["アクション1","アクション2","アクション3"]:
            if col in g.columns:
                arrs.append(g[col].values)
        if not arrs: return False
        vals = np.concatenate(arrs)
        return np.isin(vals, list(tags)).any()
    flags = (x.groupby(["試合ID","possession_id"])
               .apply(lambda g: pd.Series({
                   "is_fastbreak": has_tag_any(g, {35}),
                   "is_second_chance": has_tag_any(g, {37}),
               }))
               .reset_index())

    # ---------- 相手チーム ----------
    team_two = (x.groupby("試合ID")["チームID"]
                  .apply(lambda s: s.dropna().astype(int).value_counts().index[:2].tolist())
                  .to_dict())
    def find_opp(gid, my_team):
        L = team_two.get(gid, [])
        if len(L) == 2:
            return L[1] if int(my_team) == L[0] else L[0]
        return np.nan

    # ---------- 1ポゼッション=1行 ----------
    feat = starts_joined[[
        "試合ID","ピリオド","possession_id","possession_team",
        "score_margin_start","clock_start","x座標","y座標"
    ]].copy()
    feat = feat.rename(columns={"x座標":"x_start","y座標":"y_start"})

    feat = (feat.merge(flags, on=["試合ID","possession_id"], how="left")
                .merge(poss_points, on=["試合ID","possession_id","possession_team"], how="left"))

    feat["opponent_team"] = [find_opp(g, t) for g, t in zip(feat["試合ID"], feat["possession_team"])]

    feat = feat[[
        "試合ID","ピリオド","possession_id","possession_team",
        "score_margin_start","clock_start",
        "x_start","y_start",
        "is_fastbreak","is_second_chance","opponent_team","points_scored"
    ]].reset_index(drop=True)

    # ---------- 保存 ----------
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_CSV), exist_ok=True)
    feat.to_csv(OUTPUT_FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 出力完了: {OUTPUT_FEATURES_CSV}")
    print(feat.head(10))

    # ======================================================
    # 🆕 【追加】EPVヒートマップ描画（x,y座標を利用）
    # ======================================================
    print("\n📊 EPVヒートマップを作成中...")

    # ヒートマップ用データ
    heatmap_data = (
        feat.groupby(["x_start","y_start"])["points_scored"]
            .mean().reset_index()
    )

    # ピボット化
    heatmap_pivot = heatmap_data.pivot_table(
        index="y_start", columns="x_start", values="points_scored"
    )

    # 描画
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, cmap="RdYlGn", cbar_kws={'label': '平均得点 (EPV)'})
    plt.title("EPVヒートマップ（ポゼッション開始位置）", fontsize=14)
    plt.xlabel("x座標（コート横方向）")
    plt.ylabel("y座標（コート縦方向）")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
    
