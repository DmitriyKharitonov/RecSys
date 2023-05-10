import numpy as np
import pandas as pd
from collections import defaultdict
# from surprise import Dataset, Reader, KNNWithMeans, accuracy
# from surprise.model_selection import GridSearchCV
# from surprise.model_selection import train_test_split
# from sklearn.model_selection import train_test_split as tts
# from surprise.model_selection import KFold
# from fastai.collab import CollabDataLoaders, collab_learner
# from sklearn.preprocessing import StandardScaler
# from surprise import SVD
# from tqdm import tqdm


def fastai_predict(df, test, scaler):
	train_cdl = CollabDataLoaders.from_df(
	df,
	user_name='UID', 
	item_name='JID', 
	rating_name= "Rating",
	bs=2**14,
	valid_pct=0.1
	)

	learn = collab_learner(
		train_cdl, 
		n_factors=240, # отсылка к популярной школе в Питере
		y_range=[-10.0 ,10.0],
	)

	learn.fit_one_cycle(70, 1e-4, wd=0.2)
	return learn


def svd_predict(df, test,scaler):
	reader = Reader(rating_scale=(-10, 10))
	data = Dataset.load_from_df(df[['UID', 'JID', 'Rating']], reader)

	trainset_data = data.build_full_trainset()
	trainset, testset = train_test_split(data, test_size=0.000001)

	svd = SVD(verbose = True, n_epochs = 30,n_factors = 30)
	svd.fit(trainset)
	return svd


def full_data(learn, svd):
	df = pd.read_excel('/content/drive/MyDrive/jester-data-1.xlsx')
	df = df.replace(99, np.nan)
	df = df.rename(columns = {'Unnamed: 0' : 'UID'})

	df_arr = df.drop(columns = 'UID').to_numpy()

	test = pd.read_csv('/content/drive/MyDrive/test_joke_df_nofactrating.csv', index_col = 0)
	test_arr = test.to_numpy()

	#затрем рейтинг записей, которые попадают в тестовый набор
	for elem in test_arr:
		df_arr[elem[0]-1][elem[1]-1] = np.nan

	# приведение данных к виду (UID, JID, Rating)
	df.UID.to_numpy()
	UID = np.repeat(df.UID.to_numpy(),100)
	Rating = df_arr.flatten()
	JID =  np.tile(list(range(1,101)),len(df.UID.to_numpy()))

	full_data = pd.DataFrame(UID, columns = ["UID"])
	full_data["JID"] = JID
	full_data["Rating"] = Rating
	full_data

	full_data['SVD_predict'] = full_data[['UID', 'JID']].apply(lambda x: svd.predict(x[0], x[1], verbose=False).est, axis = 1)
	pred,_ = learn.get_preds(dl = learn.dls.test_dl(full_data[['UID', 'JID', 'Rating']]))
	full_data['Fastai_predict'] = pred
	return full_data


def prepare():
	test = pd.read_csv('/content/drive/MyDrive/test_joke_df_nofactrating.csv', index_col = 0)
	df = pd.read_csv('/content/drive/MyDrive/train_joke_df.csv')
	df = df.sort_values(by=['UID', 'JID'])
	df = df.reset_index(drop=True)

	scaler = StandardScaler()
	scaler.fit(df)
	df_scaler = scaler.transform(df)
	df['Rating'] = df_scaler[:,2]
	return df, test, scaler


def make_recomendation(full_data):
	# ranking_joke = []
 #  	for i in tqdm(test.UID.unique()):
	#     svd_top = full_data[full_data['UID'] == i+1][['JID', 'SVD_predict']].sort_values(by = 'SVD_predict', ascending = False)[:10].rename(columns = {'SVD_predict' : 'Rating'})
	#     fastai_top = full_data[full_data['UID'] == i+1][['JID', 'Fastai_predict']].sort_values(by = 'Fastai_predict', ascending = False)[:10].rename(columns = {'Fastai_predict' : 'Rating'})
	#     real_top = full_data[full_data['UID'] == i+1][['JID', 'Rating']].sort_values(by = 'Rating', ascending = False)[:10]
    
 #    ranking_joke.append(pd.concat([svd_top, fastai_top, real_top]).drop_duplicates()[:10].JID.to_list())

 #    for i in range(len(ranking_joke)):
 #  		if len(ranking_joke[i]) != 10:
 #  			ranking_joke[i] = ranking_joke[0]

	test = pd.read_csv('../Data/test_joke_df_nofactrating.csv')
	res = np.load('../Data/recomend_for_test.npy', allow_pickle=True)
	res_list = res.tolist() 
	answ = np.zeros((len(res_list), 2), dtype = list)

	for i in range(len(res_list)):
		answ[i][0] = [res_list[i][0]]
		answ[i][1] = res_list[i]

	UID = [i for i in range(1,24984)]

	UID_df = pd.read_csv('../Data/input.csv')
	recomend_df = pd.DataFrame({'UID' : UID, 'Recomend' : answ.tolist()})

	UID_df = UID_df.merge(recomend_df, how = 'inner', on = 'UID')
	UID_df = UID_df.set_index('UID')
	UID_df.to_csv('../Data/output.csv')


def main():
	# df, test, scaler = prepare()
	# learn = fastai_predict(df, test, scaler)
	# svd = svd_predict(df, test, scaler)
	# full_data = full_data(learn, svd)

	make_recomendation(full_data)

if __name__ == "__main__":
	main()