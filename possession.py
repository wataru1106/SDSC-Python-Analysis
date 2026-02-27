import pandas as pd
import numpy as np

df = pd.read_csv("/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/koshigaya_all_opponent.csv")


def label_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    両チーム混在のプレイバイプレイ df にポゼッションを付与します。
    期待する列: '試合ID','ピリオド','履歴No','チームID','アクション1','アクション2','アクション3'
    付与する列: 'possession_id','possession_team','possession_start_flag','possession_end_flag'
    """

    # ---- 1) 前処理：型・並び順 ----
    use_cols = ["試合ID","ピリオド","履歴No","チームID","アクション1","アクション2","アクション3"]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が見つかりません: {missing}")

    x = df.copy()
    for c in ["試合ID","ピリオド","履歴No","チームID","アクション1","アクション2","アクション3"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype("Int64")

    x = x.sort_values(["試合ID","ピリオド","履歴No"], ascending=[True, True, True]).reset_index(drop=True)

    # ---- 2) アクション集合 ----
    MADE_FG = {1, 3, 4, 44}
    MISSED_FG = {2, 5, 6, 45}
    OREB = {10, 18}
    DREB = {9, 19}
    FT = {7, 8}
    STEAL = {14}

    OFF_FOUL_MAIN = {23}
    TURNOVER_MAIN = {13, 17}
    TURNOVER_KINDS = {147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,163}
    SHOT_CLOCK = {34, 156}  # 156 は TO 詳細にも含まれる流派あり

    # 中立イベント：交代/TO/レビュー/開始終了など（保持判定は次の有意味イベントで行う）
    NEUTRAL = {
        80,81,82,83,84,85,86,87,88,89,  # 試合/ピリオド開始・終了、時計start/stop、交代、TO
        90,107,108,109,110,111,         # オフィシャル、ジャンプボール関連
        133,134,135,136,137,138,        # admin系
        139,140,141,142,143,144,        # タイムアウトの種類/レビュー/メディア等
        116,117,118,112,113,114,115     # アウトオブバウンズ等（次行で保持が決まる想定のタグ群）
    }

    # ---- 3) 出力列を用意 ----
    x["possession_id"] = pd.NA
    x["possession_team"] = pd.NA
    x["possession_start_flag"] = 0
    x["possession_end_flag"] = 0

    # ---- 4) 本体：試合ごとにステートマシンで走査 ----
    # gidには試合ID、sub_idxにはデータフレームxでのインデックスが返される　
    for gid, sub_idx in x.groupby("試合ID").groups.items():
        idxs = list(sub_idx)

        current_team = None
        poss_id = 0
        waiting_rebound = False
        in_ft_sequence = False
    #　possession_idとpossession_teamを決定し、possessionのフラグを立てる
        def start(team, i):
            nonlocal poss_id, current_team, waiting_rebound, in_ft_sequence
            poss_id += 1
            current_team = int(team)
            waiting_rebound = False
            in_ft_sequence = False
            x.loc[i, "possession_id"] = poss_id
            x.loc[i, "possession_team"] = current_team
            x.loc[i, "possession_start_flag"] = 1

        def carry(i):
            x.loc[i, "possession_id"] = poss_id
            x.loc[i, "possession_team"] = current_team

        def end(i):
            x.loc[i, "possession_end_flag"] = 1

        for i in idxs:
            a1 = x.at[i, "アクション1"]
            a2 = x.at[i, "アクション2"]
            a3 = x.at[i, "アクション3"]
            t  = x.at[i, "チームID"]

            code1 = int(a1) if pd.notna(a1) else None
            team  = int(t) if pd.notna(t) else None

            # ユーティリティ
            def is_neutral(c):   return (c is None) or (c in NEUTRAL)
            def is_ft(c):        return (c in FT)
            def is_made_fg(c):   return (c in MADE_FG)
            def is_miss_fg(c):   return (c in MISSED_FG)
            def is_oreb(c):      return (c in OREB)
            def is_dreb(c):      return (c in DREB)
            def is_steal(c):     return (c in STEAL)
            def is_to_like(c):   return (c in TURNOVER_MAIN) or (c in TURNOVER_KINDS) or (c in SHOT_CLOCK) or (c in OFF_FOUL_MAIN)

            # ---- まだポゼ未確定 ----
            if pd.isna(current_team):
                if code1 in DREB or code1 in STEAL:
                    start(team, i)
                    # ミス直後・FT連続のフラグは通常ここでは立たないが整合性のため
                    if is_miss_fg(code1): waiting_rebound = True
                    if is_ft(code1) and team == current_team: in_ft_sequence = True
                    if team == current_team and (is_made_fg(code1) or is_to_like(code1)):
                        end(i); current_team = pd.NA
                    continue
                elif code1 in (MADE_FG | MISSED_FG | FT | OREB | TURNOVER_MAIN | OFF_FOUL_MAIN | TURNOVER_KINDS | SHOT_CLOCK):
                    # 攻撃側しか起きない明確な行為 → この行のチームで開始
                    start(team, i)
                    if is_miss_fg(code1): waiting_rebound = True
                    if is_ft(code1) and team == current_team: in_ft_sequence = True
                    if team == current_team and (is_made_fg(code1) or is_to_like(code1)):
                        end(i); current_team = pd.NA
                    continue
                else:
                    # 中立イベントはスキップ
                    continue

            # ---- ここから current_team が有効 ----
            carry(i)

            # ミス後のリバウンド待ち
            if waiting_rebound:
                if team == current_team and is_oreb(code1):
                    waiting_rebound = False               # OREB→継続
                elif team != current_team and (is_dreb(code1) or is_steal(code1)):
                    end(i); current_team = pd.NA          # DREB/STEAL→交代
                    start(team, i)
                elif is_neutral(code1):
                    pass                                  # 中立は読み飛ばす
                elif team != current_team:
                    end(i); current_team = pd.NA          # 相手の攻撃行為→交代
                    start(team, i)
                else:
                    waiting_rebound = False               # まれ：同チームの別行為→継続
                continue

            # FT 連続中
            if in_ft_sequence:
                if team == current_team and is_ft(code1):
                    # 連続中は継続
                    pass
                else:
                    # 連続終了 → 次イベントで保持判定
                    if team == current_team and is_oreb(code1):
                        in_ft_sequence = False             # OREB→継続
                    elif team != current_team and (is_dreb(code1) or is_steal(code1)):
                        end(i); current_team = pd.NA       # DREB/STEAL→交代
                        start(team, i)
                    elif is_neutral(code1):
                        pass                               # 中立は読み飛ばす
                    elif team != current_team:
                        end(i); current_team = pd.NA       # 相手の攻撃行為→交代
                        start(team, i)
                    else:
                        in_ft_sequence = False             # 同チームの別行為→継続
                continue

            # 通常状態
            if team == current_team and is_made_fg(code1):
                end(i); current_team = pd.NA
            elif team == current_team and is_to_like(code1):
                end(i); current_team = pd.NA
            elif team == current_team and is_miss_fg(code1):
                waiting_rebound = True
            elif team == current_team and is_ft(code1):
                in_ft_sequence = True
            elif team != current_team and (is_dreb(code1) or is_steal(code1)):
                end(i); current_team = pd.NA
                start(team, i)

        # ピリオド/試合の終端で未終了ポゼッションが残っていれば、最後の行に終了フラグを打つ（任意）
        # これを入れると per-game の両チームポゼッション数差が安定します。
        # 最後が中立行で終わる場合もあるので、ゲーム末尾から遡って最初の「保持に関係する行」を探す実装も可。

    # ---- 5) 中立行にも posession_id/team を前方埋め（見やすさのため） ----
    def ffill_within_game(sub):
        sub["possession_id"] = sub["possession_id"].ffill()
        sub["possession_team"] = sub["possession_team"].ffill()
        return sub

    x = x.groupby("試合ID", group_keys=False).apply(ffill_within_game)

    return x

# 使い方：
# df はあなたのプレイバイプレイ DataFrame
possession_df = label_possessions(df)
possession_df.to_csv('/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/possession_df.csv')

# 例：越谷の攻撃ポゼッションだけに絞る
# KOSHI_ID = 745  # 必要に応じて置き換え
# koshi_poss = df[df["possession_team"] == KOSHI_ID].copy()
