import pandas as pd
from typing import Optional, Dict, Set

df = pd.read_csv("/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/koshigaya_all_opponent.csv")

def label_possessions_with_row_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    プレイバイプレイに「ポゼッションID」「開始行」「終了行」を付与する。
    重要な仕様：
      - ポゼ境界は「行インデックス」で厳密に管理
        * 同一行で交代が起きた場合：前ポゼ.end = i / 次ポゼ.start = i を両立（フラグの競合なし）
      - 行の possession_id は「その行が属するポゼ」にのみ付与（ポゼ外行はNA）
        * これにより ffill で誤って塗り伸ばされる問題を回避（and-oneのFT等は無所属でOK）
      - 「終了直後の中立（スローイン等）」でも新ポゼを開始できる
        * team付きの中立 → そのteamで開始
        * team無しの中立 → 直前に閉じたチームの“相手チーム”で開始（試合内2チームが取得できる場合）
      - A1/A2/A3 のいずれかに該当すればイベント成立と判定

    付与列：
      - possession_id（Int64）：行が属するポゼID（ポゼ外はNA）
      - possession_team（Int64）：当該ポゼのチームID（ポゼ外はNA）
      - possession_start_row_index（Int64）：そのポゼの開始行インデックス
      - possession_end_row_index（Int64）：そのポゼの終了行インデックス
    """

    # ---- 入力検査 & ソート（処理順は試合ID・ピリオド・履歴Noの昇順）----
    need = ["試合ID","ピリオド","履歴No","チームID","アクション1","アクション2","アクション3"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"必要列が見つかりません: {miss}")

    x = df.copy()
    for c in need:
        # 数値化（文字が混じっても NaN にして落とさない）
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.sort_values(["試合ID","ピリオド","履歴No"], ascending=True).reset_index(drop=True)

    # ---- アクション集合（マスタに合わせて調整）----
    MADE_FG: Set[int]     = {1, 3, 4, 44}
    MISSED_FG: Set[int]   = {2, 5, 6, 45}
    OREB: Set[int]        = {10, 18}
    DREB: Set[int]        = {9, 19}
    FT: Set[int]          = {7, 8}
    STEAL: Set[int]       = {14}

    OFF_FOUL_MAIN: Set[int]  = {23}
    TURNOVER_MAIN: Set[int]  = {13, 17}
    TURNOVER_KINDS: Set[int] = {147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,163}
    SHOT_CLOCK: Set[int]     = {34, 156}  # 156 はTO詳細にも重複しうる

    # 中立イベント（clock,交代,TO,レビュー等）
    NEUTRAL: Set[int] = {
        80,81,82,83,84,85,86,87,88,89,     # 試合/ピリオド開始・終了/clock/交代
        90,107,108,109,110,111,            # オフィシャル、ジャンプボール関連など
        133,134,135,136,137,138,           # 管理用
        139,140,141,142,143,144,           # タイムアウト/レビュー/メディア等
        116,117,118,112,113,114,115        # OOB/スローイン等（再開系）
    }
    # 中立だが「ポゼ開始に使ってよい（再開系）」コード
    NEUTRAL_CAN_START: Set[int] = {112,113,114,115,116,117,118}

    # ---- 判定ユーティリティ（A1/A2/A3のどれか一致で True）----
    def any_in(codes: Set[int], a1, a2, a3) -> bool:
        for a in (a1, a2, a3):
            if pd.notna(a):
                try:
                    if int(a) in codes:
                        return True
                except Exception:
                    pass
        return False

    def is_neutral(a1,a2,a3):            # 全NAも中立扱い
        return (pd.isna(a1) and pd.isna(a2) and pd.isna(a3)) or any_in(NEUTRAL,a1,a2,a3)
    def is_neutral_can_start(a1,a2,a3):  return any_in(NEUTRAL_CAN_START,a1,a2,a3)

    def is_ft(a1,a2,a3):     return any_in(FT,a1,a2,a3)
    def is_made(a1,a2,a3):   return any_in(MADE_FG,a1,a2,a3)
    def is_miss(a1,a2,a3):   return any_in(MISSED_FG,a1,a2,a3)
    def is_oreb(a1,a2,a3):   return any_in(OREB,a1,a2,a3)
    def is_dreb(a1,a2,a3):   return any_in(DREB,a1,a2,a3)
    def is_steal(a1,a2,a3):  return any_in(STEAL,a1,a2,a3)
    def is_tolike(a1,a2,a3): return any_in(TURNOVER_MAIN.union(TURNOVER_KINDS, SHOT_CLOCK, OFF_FOUL_MAIN), a1,a2,a3)

    # ---- 出力列（行単位の所属のみ塗る：ポゼ外はNAのまま）----
    x["possession_id"] = pd.NA
    x["possession_team"] = pd.NA
    x["possession_start_row_index"] = pd.NA
    x["possession_end_row_index"] = pd.NA

    # ポゼID→開始/終了インデックス
    poss_start: Dict[int, int] = {}
    poss_end: Dict[int, int]   = {}

    # ---- 試合ごとに走査 ----
    for gid, sub_idx in x.groupby("試合ID").groups.items():
        idxs = list(sub_idx)  # すでに x は全体でソート済みなので、indexの昇順＝時間順

        # 試合に出現する2チーム（NA除く）を特定し“相手チーム”を導く
        teams_in_game = [int(t) for t in x.loc[idxs, "チームID"].dropna().astype(int).unique().tolist()]
        teams_in_game = teams_in_game[:2]  # 念のため2つに制限
        def opponent_of(team: Optional[int]) -> Optional[int]:
            if team is None or len(teams_in_game) != 2:
                return None
            return teams_in_game[1] if teams_in_game[0] == team else teams_in_game[0] if team in teams_in_game else None

        # 状態
        current_team: Optional[int] = None     # 現在の攻撃側（None＝ポゼ外）
        poss_id = 0
        waiting_reb = False                    # FGミス直後のリバウンド待ち
        in_ft_seq   = False                    # 連続FT中
        pending_open = False                   # 直前にポゼを閉じて“開始待ち”状態
        prev_team_at_close: Optional[int] = None  # 直前に閉じたポゼのチーム

        # 操作関数
        def open_possession(team: int, row_i: int) -> None:
            """新ポゼを row_i から開始。行自体も新ポゼで塗る。"""
            nonlocal poss_id, current_team, waiting_reb, in_ft_seq, pending_open, prev_team_at_close
            poss_id += 1
            current_team = int(team)
            waiting_reb = False
            in_ft_seq   = False
            pending_open = False
            prev_team_at_close = None
            poss_start[poss_id] = row_i
            x.loc[row_i, "possession_id"] = poss_id
            x.loc[row_i, "possession_team"] = current_team

        def paint_current(row_i: int) -> None:
            """現ポゼの所属を row_i に塗る（境界メタは触らない）。"""
            x.loc[row_i, "possession_id"] = poss_id
            x.loc[row_i, "possession_team"] = current_team

        def close_current_at(row_i: int) -> None:
            """現ポゼを row_i で終了（行の所属は塗り替えない）。"""
            nonlocal current_team, pending_open, prev_team_at_close
            poss_end[poss_id] = row_i
            prev_team_at_close = current_team
            current_team = None
            pending_open = True  # 以降、相手側の行（中立でも）で開ける

        # 本体ループ（時間順）
        for i in idxs:
            a1, a2, a3 = x.at[i, "アクション1"], x.at[i, "アクション2"], x.at[i, "アクション3"]
            t          = x.at[i, "チームID"]
            team: Optional[int] = int(t) if pd.notna(t) else None

            # 1) 終了直後の“開始待ち”：
            #    - 相手チームの行が来ればイベント種別に関わらず開く（スローイン等の中立もOK）
            #    - teamが無いが「再開系中立」の場合、相手チームが特定できればそれで開く
            if current_team is None and pending_open:
                if team is not None and (prev_team_at_close is None or team != prev_team_at_close):
                    open_possession(team, i)
                    continue
                elif team is None and is_neutral_can_start(a1, a2, a3):
                    nxt = opponent_of(prev_team_at_close)
                    if nxt is not None:
                        open_possession(nxt, i)
                        continue
                # それ以外（純adminなど）はスキップ
                # ※ ここで continue しないのは、下の「未確定通常処理」にも回したいケースがあるため

            # 2) まだ誰のポゼでもない通常時（ゲーム開始直後など）：
            if current_team is None:
                # 明確な獲得（DREB/STEAL）で開始
                if team is not None and (is_dreb(a1,a2,a3) or is_steal(a1,a2,a3)):
                    open_possession(team, i)
                # team付きの「再開系中立」（インバウンズ等）でも開始可
                elif team is not None and is_neutral_can_start(a1,a2,a3):
                    open_possession(team, i)
                # （任意）攻撃行為から開いてもよい場合は、以下を有効化：
                # elif team is not None and (is_made(a1,a2,a3) or is_miss(a1,a2,a3) or is_ft(a1,a2,a3) or is_oreb(a1,a2,a3) or is_tolike(a1,a2,a3)):
                #     open_possession(team, i)
                else:
                    # 中立や不明瞭な行はスキップ（ポゼ外のNAを維持）
                    pass
                continue

            # 3) 現ポゼあり：まず行を現ポゼで塗る
            paint_current(i)

            # 3a) ミス後のREB待ち
            if waiting_reb:
                if team == current_team and is_oreb(a1,a2,a3):
                    waiting_reb = False  # 自軍OREB → 継続
                elif team != current_team and (is_dreb(a1,a2,a3) or is_steal(a1,a2,a3)):
                    # 交代（同行で end & start 両立）
                    close_current_at(i)
                    open_possession(team, i)
                    waiting_reb = False
                    in_ft_seq   = False
                elif is_neutral(a1,a2,a3):
                    # 時計/交代/TO等 → 所有は変えない
                    pass
                elif team != current_team:
                    # 相手の攻撃行為（保持を示唆）→ 交代（同行end/start）
                    close_current_at(i)
                    open_possession(team, i)
                    waiting_reb = False
                    in_ft_seq   = False
                else:
                    # 同チームの別行為 → 継続
                    waiting_reb = False
                continue

            # 3b) 連続FT中
            if in_ft_seq:
                if team == current_team and is_ft(a1,a2,a3):
                    # 継続（まだボールはライブでない）
                    pass
                else:
                    # 連続終了、次イベントで保持判定
                    if team == current_team and is_oreb(a1,a2,a3):
                        in_ft_seq = False  # 自軍OREB → 継続
                    elif team != current_team and (is_dreb(a1,a2,a3) or is_steal(a1,a2,a3)):
                        close_current_at(i)
                        open_possession(team, i)
                        in_ft_seq = False
                    elif is_neutral(a1,a2,a3):
                        pass
                    elif team != current_team:
                        close_current_at(i)
                        open_possession(team, i)
                        in_ft_seq = False
                    else:
                        in_ft_seq = False
                continue

            # 3c) 通常状態
            if team == current_team and is_made(a1,a2,a3):
                # FG成功はポゼ終了。and-one のFTは次行以降の“無所属区間”で処理される
                close_current_at(i)
            elif team == current_team and is_tolike(a1,a2,a3):
                close_current_at(i)
            elif team == current_team and is_miss(a1,a2,a3):
                waiting_reb = True
            elif team == current_team and is_ft(a1,a2,a3):
                in_ft_seq = True
            elif team != current_team and (is_dreb(a1,a2,a3) or is_steal(a1,a2,a3)):
                # 相手が明確に獲得 → 同行 end/start
                close_current_at(i)
                open_possession(team, i)
            else:
                # 中立・その他 → 継続
                pass

        # 4) 試合末尾：未クローズが残っていれば、その試合の最後の行で閉じる
        if current_team is not None:
            poss_end[poss_id] = idxs[-1]

    # ---- possessionごとの start/end を各行へブロードキャスト ----
    # （possession_id が付いた行にのみ付与。ポゼ外行のNAはそのまま残す）
    pid = x["possession_id"].astype("Int64")
    x["possession_start_row_index"] = pid.map({k:int(v) for k,v in poss_start.items()})
    x["possession_end_row_index"]   = pid.map({k:int(v) for k,v in poss_end.items()})

    # 注意：ここでは ffill を行わない。
    #  ポゼ外（終了〜開始の間）の行を NA のまま残すことで、
    #  「and-oneのFT」「得点後の純中立」等を“無所属”として識別できる。
    #  もし可読性のために塗りたい場合は、"完全にポゼ内に含まれる中立行"だけを条件付きで埋めること。

    return x

possession_df = label_possessions_with_row_index(df)
possession_df.to_csv('/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/possession_df_ver2.csv')
