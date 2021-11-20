import pandas as pd

# ! Last Pseudo Labeling
fold_0 = pd.read_csv('')
# fold_1 = pd.read_csv('')
fold_2 = pd.read_csv('')
# fold_3 = pd.read_csv('')
fold_4 = pd.read_csv('')

fold_1 = pd.read_csv('')
fold_3 = pd.read_csv('')

df = pd.concat([fold_0,fold_1,fold_2,fold_3,fold_4])
# df = pd.concat([fold_0,fold_2,fold_4])
df = df.groupby('uid').mean()

df['_max'] = df.max(axis=1)
df['disease_code'] = df.idxmax(axis=1).str[-1].astype(int)

# df[['disease_code']].to_csv('deterministic_re_pseudo_labeling_no_tta.csv')

pseudo_label_df = df[df['_max'] > 0.85][['disease_code']] # 4211 len
# #! 0.7 
pseudo_label_df['img_path'] = 'test_imgs/'+pseudo_label_df.index.astype(str)+'.jpg'
pseudo_label_df = pseudo_label_df.reset_index()[['uid','img_path','disease_code']]
org_train = pd.read_csv('fd_data/train.csv')

full_train_df = pd.concat([org_train[['uid','img_path','disease_code']], pseudo_label_df],axis=0)
full_train_df.to_csv('full_0.85_train.csv',index=False)
