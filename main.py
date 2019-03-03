#! /usr/bin/env python
# -*- coding: utf-8 -*-

#Импорт системных библиотек
import sys
import os
import math

import mysql.connector

#Импорт библиотеки nltk
import nltk
from nltk.corpus import stopwords

#Импорт gensim work2vec
import gensim

#Импорт numpy для dataframes
import numpy as np

import pandas.io.sql as psql

#Импорт библиотеки для работы с векторами
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#Импорт коллекций
from collections import defaultdict
import collections
from collections import Counter


#Вычисление среднего вектора
def avg_feature_vector(words, model, num_features, index2word_set, if_idf = 1):

    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            if if_idf != 1:
                new_word = np.multiply(model[index2word_set[word]], if_idf[word])
            else:
            	new_word = model[index2word_set[word]]
            featureVec = np.add(featureVec,new_word)
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)

    return featureVec

#Вычисление меры TF
def compute_tf(text):
    #На вход берем текст в виде списка (list) слов
    #Считаем частотность всех терминов во входном массиве с помощью 
    #метода Counter библиотеки collections
    tf_text = collections.Counter(text)
    for i in tf_text:
        #для каждого слова в tf_text считаем TF путём деления
        #встречаемости слова на общее количество слов в тексте
        tf_text[i] = tf_text[i]/float(len(text))
    #возвращаем объект типа Counter c TF всех слов текста
    return tf_text

#Вычисление меры idf
def compute_idf(word, corpus):
    #на вход берется слово, для которого считаем IDF
    #и корпус документов в виде списка списков слов
    #количество документов, где встречается искомый термин
    #считается как генератор списков
    return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))

#Вычисление меры IF IDF
def compute_tfidf(corpus):

    documents_list = []

    for text in corpus:

        tf_idf_dictionary = {}

        computed_tf = compute_tf(text)

        for word in computed_tf:

            value = computed_tf[word] * compute_idf(word, corpus)

            if value == 0:
                tf_idf_dictionary[word] = 1.0
            else:
                tf_idf_dictionary[word] = value

        documents_list.append(tf_idf_dictionary)

    return documents_list

#Токенизация текста
def w2v_tokenize_text(text, language_default='russian'):
	tokens = []
    # производим токенизацию текста (получаются отдельные предложения)
	for sent in nltk.sent_tokenize(text, language = language_default):
        # производим токенизацию предложений (получаются отдельные слова)
		for word in nltk.word_tokenize(sent, language = language_default):
			if len(word) < 2:
				continue
			if word in stopwords.words(language_default):
			    continue
			tokens.append(word)
	return tokens

def save_result(program_id, vacancy_id, similarity):

    cnxn = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='univer')

    cursor = cnxn.cursor()

    query = ("SELECT id FROM result "
         "WHERE program_id = %s AND vacancy_id = %s")

    cursor.execute(query, (int(program_id), int(vacancy_id)))

    if cursor.fetchone() is None:
        cursor_insert = cnxn.cursor()
        add_result = ("INSERT INTO result "
               "(program_id, vacancy_id, similarity) "
               "VALUES (%s, %s, %s)")
        cursor_insert.execute(add_result, (int(program_id), int(vacancy_id), float(similarity)))
        cnxn.commit()
        cursor_insert.close()
    
    cursor.close()
    cnxn.close()


###################################################  SQL TO DATAFRAMES  ######################################################################

clear = lambda: os.system('cls')
clear()

cnxn = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='univer')
cursor = cnxn.cursor()

teaching_programms  = psql.read_sql("SELECT * FROM teaching_programms", cnxn)

vacancies           = psql.read_sql("SELECT * FROM vacancies LIMIT 100", cnxn)

cnxn.close()

################################################################## DEFINE ARRAY FOR T.P CORPUS AND ARRAY FOR WORD2VEC'S RUSSIAN WORDS ###################################

main_corpus             = []
index2word_set_global   = {}

############################################################## CALCULATE TF_IDF VALUE FOR TEACHING PROGRAMMS ########################################

for index, row in teaching_programms.iterrows():
    main_corpus.append(w2v_tokenize_text(row['text']))

if_idf_programms = compute_tfidf(main_corpus)

##############################################################################################################################################

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('models/news_upos_cbow_300_2_2017.bin.gz', binary = True, limit=500000)

index2word_set = set(word2vec_model.index2word)

for i_set in index2word_set:
    index2word_set_global[i_set.split('_', 1)[0]] = i_set

##########################################################################################################################################################################################################################################################################

for specialization_id in (vacancies['specialization_id'].unique().tolist()):

    specialized_vacansies   = vacancies.loc[vacancies['specialization_id'] == specialization_id]
    vacancies_corpus        = []

    for index_v, row in specialized_vacansies.iterrows():
        vacancies_corpus.append(w2v_tokenize_text(row['text']))

    if_idf_vacancies = compute_tfidf(vacancies_corpus)
    
    for index, corpus in enumerate(vacancies_corpus):

        sentence_1 = corpus
        sentence_1_avg_vector = avg_feature_vector(sentence_1, model=word2vec_model, num_features=300, index2word_set=index2word_set_global, if_idf=if_idf_vacancies[index])

        for index_p, corpus_p in enumerate(main_corpus):

            sentence_2 = corpus_p
            sentence_2_avg_vector = avg_feature_vector(sentence_2, model=word2vec_model, num_features=300, index2word_set=index2word_set_global, if_idf=if_idf_programms[index_p])

            sentence_1_avg_vector = sentence_1_avg_vector.reshape(1, -1)
            sentence_2_avg_vector = sentence_2_avg_vector.reshape(1, -1)

            save_result(teaching_programms.iloc[index_p]['id'], specialized_vacansies.iloc[index]['id'], cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)[0][0])

            print('Programm id:',  teaching_programms.iloc[index_p]['id'], 'Vacancy id:', specialized_vacansies.iloc[index]['id'], 'Similarity:',cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)[0][0])

##############################################################################################################################################################################################################################################################################