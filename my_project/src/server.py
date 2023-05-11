import hashlib
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from flask import Flask, render_template, request, redirect, url_for
from .config import *
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split as tts
from surprise.model_selection import KFold
from fastai.collab import CollabDataLoaders, collab_learner
from sklearn.preprocessing import StandardScaler
from surprise import SVD

# Создаем логгер и отправляем информацию о запуске
# Важно: логгер в Flask написан на logging, а не loguru,
# времени не было их подружить, так что тут можно пересоздать 
# logger из logging
logger.add(LOG_FOLDER + "log.log")
logger.info("Наш запуск")

# Создаем сервер и убираем кодирование ответа
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  


@app.route("/<task>")
def main(task: str):
    """
    Эта функция вызывается при вызове любой страницы, 
    для которой нет отдельной реализации

    Пример отдельной реализации: add_data
    
    Параметры:
    ----------
    task: str
        имя вызываемой страницы, для API сделаем это и заданием для сервера
    """
    return render_template('index.html', task=task)


@app.route("/add_data", methods=['POST'])
def upload_file():
    """
    Страница на которую перебросит форма из main 
    Здесь происходит загрузка файла на сервер
    """
    def allowed_file(filename):
        """ Проверяем допустимо ли расширение загружаемого файла """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'add_data'

    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        answer['Сообщение'] = 'Нет файла'
        return answer
    file = request.files['file']

    # Проверяем что путь к файлу не пуст
    if file.filename == '':
        answer['Сообщение'] = 'Файл не выбран'
        return answer
    
    # Загружаем
    if file and allowed_file(file.filename):
        filename = hashlib.md5(file.filename.encode()).hexdigest() 
        file.save(
            os.path.join(
                UPLOAD_FOLDER, 
                filename + file.filename[file.filename.find('.'):]
                )
            )
        answer['Сообщение'] = 'Файл успешно загружен!'
        answer['Успех'] = True
        answer['Путь'] = filename
        return answer
    else:
        answer['Сообщение'] = 'Файл не загружен'
        return answer
  

@app.route("/show_data", methods=['GET'])
def show_file():
    """
    Страница выводящая содержимое файла
    """
   
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'show_file'

    # Проверяем, что указано имя файла
    if 'path' not in request.args:
        answer['Сообщение'] = 'Не указан путь файла'
        return answer
    file = request.args.get('path') 
    
    # Проверяем, что указан тип файла
    if 'type' not in request.args:
        answer['Сообщение'] = 'Не указан тип файла'
        return answer
    type = request.args.get('type')

    file_path = os.path.join(UPLOAD_FOLDER, file + '.' + type)

    # Проверяем, что файл есть
    if not os.path.exists(file_path):
        answer['Сообщение'] = 'Файл не существует'
        return answer

    answer['Сообщение'] = 'Файл успешно загружен!'
    answer['Успех'] = True
    
    # Приводим данные в нужный вид
    if type == 'csv':
        answer['Данные'] = pd.read_csv(file_path).to_dict()
        return answer
    else:
        answer['Данные'] = 'Не поддерживаемы тип'
        return answer
    

@app.route("/start", methods=['POST'])
def start_model():
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
        #файл с полным дасетом
        df = pd.read_excel('./files/jester-data-1.xlsx')
        df = df.replace(99, np.nan)
        df = df.rename(columns = {'Unnamed: 0' : 'UID'})

        df_arr = df.drop(columns = 'UID').to_numpy()

        test = pd.read_csv('./files/test_joke_df_nofactrating.csv', index_col = 0)
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

        #предсказываем рейтинг
        full_data['SVD_predict'] = full_data[['UID', 'JID']].apply(lambda x: svd.predict(x[0], x[1], verbose=False).est, axis = 1)
        pred,_ = learn.get_preds(dl = learn.dls.test_dl(full_data[['UID', 'JID', 'Rating']]))
        full_data['Fastai_predict'] = pred
        return full_data


    def prepare():

        #df и test - тренировочный набор из первого этапа. 
        test = pd.read_csv('./files/test_joke_df_nofactrating.csv', index_col = 0)
        df = pd.read_csv('./files/train_joke_df.csv')
        df = df.sort_values(by=['UID', 'JID'])
        df = df.reset_index(drop=True)

        scaler = StandardScaler()
        scaler.fit(df)
        df_scaler = scaler.transform(df)
        df['Rating'] = df_scaler[:,2]
        return df, test, scaler


    def make_recomendation(full_data, test):

        top_recomend = []
        ranking_joke = []

        #берем 10 шуток с наибольшим предсказанным рейтингом
        for i in test.UID.unique():
            svd_top = full_data[full_data['UID'] == i+1][['JID', 'SVD_predict']].sort_values(by = 'SVD_predict', ascending = False)[:10].rename(columns = {'SVD_predict' : 'Rating'})
            fastai_top = full_data[full_data['UID'] == i+1][['JID', 'Fastai_predict']].sort_values(by = 'Fastai_predict', ascending = False)[:10].rename(columns = {'Fastai_predict' : 'Rating'})
            real_top = full_data[full_data['UID'] == i+1][['JID', 'Rating']].sort_values(by = 'Rating', ascending = False)[:10]

            top_recomend.append(pd.concat([svd_top, fastai_top, real_top]).drop_duplicates()[:1].Rating)
            ranking_joke.append(pd.concat([svd_top, fastai_top, real_top]).drop_duplicates()[:10].JID.to_list())

        #почему-то одна строчка остается пустой
        for i in range(len(ranking_joke)):
            if len(ranking_joke[i]) != 10:
                ranking_joke[i] = ranking_joke[0]
                top_recomend[i] = top_recomend[0]

        #test = pd.read_csv('./files/test_joke_df_nofactrating.csv')

        #res = np.load('./files/recomend_for_test.npy', allow_pickle=True)
        #res_list = res.tolist() 

        res_list = ranking_joke 
        answ = np.zeros((len(res_list), 2), dtype = list)

        for i in range(len(res_list)):
            answ[i][0] = {res_list[i][0] : top_recomend[i]}
            answ[i][1] = res_list[i]

        UID = [i for i in range(1,24984)]

        UID_df = pd.read_csv('../data/input.csv')
        print(UID_df)
        recomend_df = pd.DataFrame({'UID' : UID, 'Recomend' : answ.tolist()})

        UID_df = UID_df.merge(recomend_df, how = 'inner', on = 'UID')
        UID_df = UID_df.set_index('UID')
        UID_df[['UID', 'Recomend']].to_csv('../data/output.csv')


    df, test, scaler = prepare()
    learn = fastai_predict(df, test, scaler)
    svd = svd_predict(df, test, scaler)
    full_data = full_data(learn, svd)

    make_recomendation(full_data, test)
