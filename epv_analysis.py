import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ========= ファイル読み込み =========
df = pd.read_csv("/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/epv_features_with_xy.csv")

# ========= 越谷アルファーズのみ抽出 =========
KOSHI_ID = 745
df_koshi = df[df["possession_team"] == KOSHI_ID].copy()# ========= clock_start を数値化（エラー行の前に追加） =========
df_koshi["clock_start"] = pd.to_numeric(df_koshi["clock_start"], errors="coerce")


# ========= 欠損・型整備 =========
df_koshi = df_koshi.dropna(subset=["points_scored"])
df_koshi["is_fastbreak"] = df_koshi["is_fastbreak"].astype(bool)
df_koshi["is_second_chance"] = df_koshi["is_second_chance"].astype(bool)

# ========= 区分化 =========
df_koshi["margin_bin"] = pd.cut(df_koshi["score_margin_start"], bins=[-50,-10,-3,3,10,50], right=False)
df_koshi["clock_bin"] = pd.cut(df_koshi["clock_start"], bins=[0,24,60,120,300,600,720], right=False)

# ========= 平均EPV集計 =========
epv_table = (
    df_koshi.groupby(["is_fastbreak","is_second_chance","margin_bin","clock_bin"], dropna=False)
    .agg(
        epv=("points_scored","mean"),
        n=("points_scored","size")
    )
    .reset_index()
)

# ========= 上位状況の確認 =========
top_conditions = epv_table.sort_values("epv", ascending=False).head(10)
print("===== 得点期待が最も高い条件Top10 =====")
print(top_conditions)

# ========= 可視化：速攻×点差別の平均得点ヒートマップ =========
pivot = epv_table.pivot_table(index="margin_bin", columns="clock_bin", values="epv")
plt.figure(figsize=(8,5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("越谷アルファーズ EPV（点差×残り時間）平均得点")
plt.xlabel("残り時間区間（秒）")
plt.ylabel("点差区間（味方 - 相手）")
plt.tight_layout()
plt.show()

# ========= 可視化：速攻 vs セットプレー平均EPV =========
plt.figure(figsize=(6,4))
sns.barplot(data=df_koshi, x="is_fastbreak", y="points_scored", estimator="mean", ci=None)
plt.title("速攻あり vs なし 平均得点（越谷アルファーズ）")
plt.xlabel("速攻ありか？")
plt.ylabel("平均得点")
plt.show()
