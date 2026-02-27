import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ========= ファイル読み込み =========
df = pd.read_csv("/Users/nakamurawataru/Documents/学校/研究室/SDSC/03.バスケ/6月送付分/【2025年度】ボックススコア.csv")

use = df[['試合ID','ピリオド区分','チームID','3P成功','3P試投','2P成功','2P試投']].copy()
use = use[use['ピリオド区分'].isin([1,2,3,4,5,6])].reset_index(drop=True)
use = use.sort_values(["試合ID","ピリオド区分","チームID"]).reset_index(drop=True)

agg = use.groupby(["試合ID",'ピリオド区分',"チームID"], as_index=False).agg(
    ThreeFGA_Sum=("3P試投","sum"),
    TwoFGA_Sum=("2P試投","sum"),
)

agg['is_OT']   = agg['ピリオド区分'].between(5, 6, inclusive='both')
agg["MINUTES"] = np.where(agg["is_OT"], 5, 10)

agg["ThreeFGA_per_min"] = agg["ThreeFGA_Sum"] / agg["MINUTES"]
agg["TwoFGA_per_min"]   = agg["TwoFGA_Sum"]   / agg["MINUTES"]

def corr_of(a, b, data):
    c = np.corrcoef(data[a], data[b])[0,1] if len(data) > 1 else np.nan
    return round(float(c), 3)

overall_raw   = corr_of("ThreeFGA_Sum","TwoFGA_Sum", agg)
overall_per_m = corr_of("ThreeFGA_per_min","TwoFGA_per_min", agg)

reg_raw   = corr_of("ThreeFGA_Sum","TwoFGA_Sum", agg[~agg["is_OT"]])
reg_per_m = corr_of("ThreeFGA_per_min","TwoFGA_per_min", agg[~agg["is_OT"]])

ot_raw    = corr_of("ThreeFGA_Sum","TwoFGA_Sum", agg[agg["is_OT"]])
ot_per_m  = corr_of("ThreeFGA_per_min","TwoFGA_per_min", agg[agg["is_OT"]])

print("\n=== Pearson r ===")
print(f"[ALL]   raw={overall_raw} / per_min={overall_per_m}")
print(f"[Q1-4]  raw={reg_raw}     / per_min={reg_per_m}")
print(f"[OT]    raw={ot_raw}      / per_min={ot_per_m}")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ========== 可視化1：散布図（重なり軽減） ==========
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=agg,
    x="ThreeFGA_Sum",
    y="TwoFGA_Sum",
    hue="is_OT",
    palette={False: "#4A90E2", True: "#F5A623"},  # カラー固定で視認性UP
    s=25,         # 点をさらに小さく
    alpha=0.55,   # やや透過
    linewidth=0
)

# 軽いジッター（整数データの重なり解消）
plt.scatter(
    agg["ThreeFGA_Sum"] + np.random.uniform(-0.15, 0.15, len(agg)),
    agg["TwoFGA_Sum"] + np.random.uniform(-0.15, 0.15, len(agg)),
    s=5, alpha=0.25, c="gray", marker="."
)

plt.title("3P vs 2P Attempts (Scatter, with Jitter)", fontsize=13)
plt.xlabel("ThreeFGA_Sum (3P Attempts)")
plt.ylabel("TwoFGA_Sum (2P Attempts)")
plt.legend(title="OT?")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ========== 可視化2：Hexbin（密度ヒートマップ） ==========
plt.figure(figsize=(8, 6))
hb = plt.hexbin(
    agg["ThreeFGA_Sum"],
    agg["TwoFGA_Sum"],
    gridsize=60,      # より細かい粒度（デフォルトは30）
    mincnt=1,
    cmap="viridis",   # 明暗コントラストが強く見やすい
    alpha=0.9
)
cb = plt.colorbar(hb)
cb.set_label("Number of periods")

plt.xlabel("ThreeFGA_Sum (3P Attempts)")
plt.ylabel("TwoFGA_Sum (2P Attempts)")
plt.title("3P vs 2P Attempts (Hexbin Density, gridsize=60)", fontsize=13)
plt.grid(alpha=0.2, linestyle="--")
plt.tight_layout()
plt.show()

