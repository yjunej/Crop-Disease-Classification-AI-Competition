import pandas as pd

#! PSEUDO
# fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
# fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
# fold_4 = pd.read_csv('')

#! FINAL
# fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
# fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
# fold_4 = pd.read_csv('')

#! RE-PSEUDO with no tta
# fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
# fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
# fold_4 = pd.read_csv('')

# #! FINAL with 0.7 pseudo labeling
# fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
# fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
# fold_4 = pd.read_csv('')

# #! SOTA + SOTA
# fold_5 = pd.read_csv('')
# fold_6 = pd.read_csv('')
# fold_7 = pd.read_csv('')
# fold_8 = pd.read_csv('')
# fold_9 = pd.read_csv('')

#! Last Pseudo Labeling
# fold_0 = pd.read_csv('')
fold_1 = pd.read_csv('')
# fold_2 = pd.read_csv('')
fold_3 = pd.read_csv('')
# fold_4 = pd.read_csv('')

#! Last Final Submit
# fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
# fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
# fold_4 = pd.read_csv('')

#! Last Phase 3 Submit
# fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
# fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
# fold_4 = pd.read_csv('')

#! Phase 1 recap
fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
fold_4 = pd.read_csv('')

df = pd.concat([fold_0,fold_1,fold_2,fold_3,fold_4])

df = df.groupby('uid').mean()
series = df.idxmax(axis=1).str[-1].astype(int)
result = pd.DataFrame(series,columns=['disease_code'])
result.to_csv('phase_1_recap_2.csv')