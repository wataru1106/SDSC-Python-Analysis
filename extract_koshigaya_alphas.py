#pandasをpdとしてimportする
import pandas as pd

#「sample.csv」を手元のデータファイル名に置き換える
df_23_24 = pd.read_csv("/Users/nakamurawataru/Documents/学校/研究室/SDSC/03.バスケ/6月送付分/【2025年度】プレイバイプレイ_23-24シーズン.csv")
df_24_25 = pd.read_csv("/Users/nakamurawataru/Documents/学校/研究室/SDSC/03.バスケ/6月送付分/【2025年度】プレイバイプレイ_24-25シーズン.csv")
df_all = pd.concat([df_23_24, df_24_25], ignore_index=True)

koshigaya_alphas_23_24 = df_23_24[df_23_24['チームID'] == 745]
koshigaya_alphas_24_25 = df_24_25[df_24_25['チームID'] == 745]
koshigaya_all = df_all[df_all['チームID'] == 745]


# koshigaya_all.to_csv('/Users/nakamurawataru/Documents/学校/研究室/スポーツデータサイエンスコンペティション/analysis/koshigaya_all.csv')

# koshigaya の試合ID一覧を取得（koshigaya_all から取る）
koshigaya_match_ids = koshigaya_all['試合ID'].unique()

# 同じ試合の全行を取得。対戦相手のみ欲しいなら (df_all['チームID'] != 745) を追加
koshigaya_all_opponent = df_all[df_all['試合ID'].isin(koshigaya_match_ids)]

koshigaya_all_opponent.to_csv('/Users/nakamurawataru/Documents/学校/研究室/SDSC/analysis/koshigaya_all_opponent.csv')
