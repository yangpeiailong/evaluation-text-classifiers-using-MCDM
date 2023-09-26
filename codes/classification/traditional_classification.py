from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re
import time
import nltk
import numpy as np
import math
import pickle
import os
# import multiprocessing
import xlwt
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model, naive_bayes, metrics, svm, neighbors, tree
import fasttext.util
from fasttext import load_model



def get_words(samples_temp):
    words_temp = []
    for line_temp in samples_temp:
        tmp_list = 0
        if isinstance(line_temp, list):
            tmp_list = line_temp
        elif isinstance(line_temp, str):
            tmp_list = list(line_temp.split(' '))
        for word in tmp_list:
            if str(word) not in words_temp:
                words_temp.append(word)
    return words_temp


# 把文档集合切割为k折交叉验证的形式
'''
doc_train_list_temp: 训练集构成的列表，每个训练集占总样本数的（1 - 1/kk）
doc_test_list_temp: 测试集构成的列表，每个测试集占总样本数的1/kk
labels_train_list_temp：与doc_train_list_temp相对应的标签集构成的列表
labels_test_list_temp：与doc_test_list_temp相对应的标签集构成的列表
'''
def k_fold_cross_validation(doc_terms_list_temp, labels_temp, kk):
    skf = StratifiedKFold(n_splits=kk)
    skf.get_n_splits(doc_terms_list_temp, labels_temp)
    doc_train_list_temp = []
    labels_train_list_temp = []
    doc_test_list_temp = []
    labels_test_list_temp = []
    for train_index, test_index in skf.split(doc_terms_list_temp, labels_temp):
        # print("TRAIN:",train_index, "TEST:", test_index)
        x_train, x_test = np.array(doc_terms_list_temp, dtype=object)[train_index], np.array(doc_terms_list_temp, dtype=object)[test_index]
        y_train, y_test = np.array(labels_temp)[train_index], np.array(labels_temp)[test_index]
        doc_train_list_temp.append(x_train.tolist())
        labels_train_list_temp.append(y_train.tolist())
        doc_test_list_temp.append(x_test.tolist())
        labels_test_list_temp.append(y_test.tolist())
    return doc_train_list_temp, labels_train_list_temp, doc_test_list_temp, labels_test_list_temp


def get_term_dict(doc_terms_list_temp):
    term_dict = {}
    for doc_terms in doc_terms_list_temp:
        for term in doc_terms:
            term_dict[term] = 1
    term_set_list = sorted(term_dict.keys())  # term set 排序后，按照索引做出字典
    term_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_dict


def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return class_dict


def stats_class_df(doc_class_list, class_dict):  # 类别DF字典
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list


def stats_term_class_df(doc_terms_list_temp, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for kk in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[kk]]
        doc_terms = doc_terms_list_temp[kk]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return term_class_df_mat


def stats_term_df(term_set, term_class_df_mat):  # 词项DF字典
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    term_score_mat = A + B
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    # term_set_fs = [term_set[index] for index in sorted_term_score_index]
    term_set_fs_temp = [(term_set[index], term_score_array[index], sorted_term_score_index[index]) for index in range(len(term_set))]
    return term_set_fs_temp


def feature_selection_ig(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat # 词汇表的大小 * 类别数量构成的矩阵 它的每个元素wij表示 词vi出现且类别为cj的文档数量
    B = np.array([(sum(x) - x).tolist() for x in A])   # 词汇表的大小 * 类别数量构成的矩阵 它的每个元素wij表示 词vi不出现且类别为cj的文档数量
    C = np.tile(class_df_list, (A.shape[0], 1)) - A # 词汇表的大小 * 类别数量构成的矩阵  Cij表示词vi不出现且类别为cj的文档数量
    N = sum(class_df_list) # 文章数量
    D = N - A - B - C
    term_df_array = np.sum(A, axis=1)

    p_t = term_df_array / N
    p_not_t = 1 - p_t
    p_c_t_mat = (A + 1E-6) / (A + B + 1E-6)
    p_c_not_t_mat = (C + 1E-6) / (C + D + 1E-6)
    p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)

    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    # term_set_fs = [term_set[index] for index in sorted_term_score_index]
    term_set_fs_temp = [
        (term_set[index], term_score_array[index], sorted_term_score_index[index]) for index in range(len(term_set))]
    return term_set_fs_temp

def feature_selection_gi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    p_c_t_mat = (A + 1E-6) / (A + B + 1E-6)
    p_t_c_mat = (A + 1E-6) / (A + C + 1E-6)
    term_score_array = np.sum(p_c_t_mat * p_c_t_mat * p_t_c_mat * p_t_c_mat, axis=1)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    # term_set_fs = [term_set[index] for index in sorted_term_score_index]
    term_set_fs_temp = [
        (term_set[index], term_score_array[index], sorted_term_score_index[index]) for index in range(len(term_set))]
    return term_set_fs_temp

def feature_selection_dfs(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C

    p_c_t_mat = (A + 1E-6) / (A + B + 1E-6)
    p_not_t_c_mat = (C + 1E-6) / (A + C + 1E-6)
    p_t_not_c_mat = (B + 1E-6) / (B + D + 1E-6)
    term_score_array = np.sum(p_c_t_mat / (p_not_t_c_mat + p_t_not_c_mat + 1), axis=1)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    # term_set_fs = [term_set[index] for index in sorted_term_score_index]
    term_set_fs_temp = [
        (term_set[index], term_score_array[index], sorted_term_score_index[index]) for index in range(len(term_set))]
    return term_set_fs_temp


def feature_selection_ece(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C

    term_df_array = np.sum(A, axis=1)
    p_t = term_df_array / N
    p_c_t_mat = (A + 1E-6) / (A + B + 1E-6)

    term_score_array = p_t * np.sum(p_c_t_mat * np.log(p_c_t_mat * N/ (A + C + 1E-6)), axis=1)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    # term_set_fs = [term_set[index] for index in sorted_term_score_index]
    term_set_fs_temp = [
        (term_set[index], term_score_array[index], sorted_term_score_index[index]) for index in range(len(term_set))]
    return term_set_fs_temp


def feature_selection_chi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C

    term_score_mat = N * (A * D - C * B) * (A * D - C * B) / ((A + C + + 1E-6) * (B + D + 1E-6) * (A + B + 1E-6) * (C + D + 1E-6))
    term_score_array = np.max(term_score_mat, axis=1)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    # term_set_fs = [term_set[index] for index in sorted_term_score_index]
    term_set_fs_temp = [
        (term_set[index], term_score_array[index], sorted_term_score_index[index]) for index in range(len(term_set))]
    return term_set_fs_temp


def feature_selection_or(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C

    p_c_t_mat = (A + 1E-6) / (A + B + 1E-6)
    p_t_not_c_mat = (B + 1E-6) / (B + D + 1E-6)
    term_score_mat = np.log((p_c_t_mat * (1 - p_t_not_c_mat) + 1E-6)/ ((1 - p_c_t_mat) * p_t_not_c_mat + 1E-6))
    term_score_array = np.max(term_score_mat, axis=1)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    # term_set_fs = [term_set[index] for index in sorted_term_score_index]
    term_set_fs_temp = [
        (term_set[index], term_score_array[index], sorted_term_score_index[index]) for index in range(len(term_set))]
    return term_set_fs_temp

# 来自论文A global-ranking local feature selection method for text categorization.pdf


def feature_selection(doc_terms_list_temp, doc_class_list, fs_method):
    time_start1 = time.time()
    term_dict = get_term_dict(doc_terms_list_temp)
    # print(doc_terms_list)
    class_dict = get_class_dict(doc_class_list)
    class_df_list = stats_class_df(doc_class_list, class_dict)
    term_class_df_mat = stats_term_class_df(doc_terms_list_temp, doc_class_list, term_dict, class_dict)
    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x: x[1])]
    term_set_fs_temp = []
    time_end2 = time.time()
    time_start2 = 0
    time_end3 = 0

    if fs_method == 'DF':
        time_start2 = time.time()
        term_set_fs_temp = stats_term_df(term_set, term_class_df_mat)
        time_end3 = time.time()
    elif fs_method == 'IG':
        time_start2 = time.time()
        term_set_fs_temp = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
        time_end3 = time.time()
    elif fs_method == 'GI':
        time_start2 = time.time()
        term_set_fs_temp = feature_selection_gi(class_df_list, term_set, term_class_df_mat)
        time_end3 = time.time()
    elif fs_method == 'DFS':
        time_start2 = time.time()
        term_set_fs_temp = feature_selection_dfs(class_df_list, term_set, term_class_df_mat)
        time_end3 = time.time()
    elif fs_method == 'ECE':
        time_start2 = time.time()
        term_set_fs_temp = feature_selection_ece(class_df_list, term_set, term_class_df_mat)
        time_end3 = time.time()
    elif fs_method == 'CHI':
        time_start2 = time.time()
        term_set_fs_temp = feature_selection_chi(class_df_list, term_set, term_class_df_mat)
        time_end3 = time.time()
    elif fs_method == 'OR':
        time_start2 = time.time()
        term_set_fs_temp = feature_selection_or(class_df_list, term_set, term_class_df_mat)
        time_end3 = time.time()
    total_time = time_end3 - time_start2 + time_end2 - time_start1
    return term_set_fs_temp, total_time


def ZeroArray(n,idx):
    zeroarray = np.zeros(n)
    zeroarray[idx] = 1
    return zeroarray


# def measures(pred, y_score_temp, labels_test_temp, n_class_temp):
#     acc = metrics.accuracy_score(labels_test_temp, pred)
#     # rec = metrics.recall_score(labels_test, pred, average='macro')
#     y = []
#     labels_set = list(set(labels_test_temp))
#     for ii in range(len(labels_test_temp)):
#         labels_test_temp[ii] = labels_set.index(labels_test_temp[ii])
#     if len(n_class_temp) > 2:
#         for ii in range(len(labels_test_temp)):
#             labels_test_temp[ii] = int(labels_test_temp[ii])
#             # print(len(n_class), int(labels_test[ii]))
#             y.append(ZeroArray(len(n_class_temp), labels_test_temp[ii]))
#     elif len(n_class_temp) == 2:
#         from sklearn.preprocessing import LabelBinarizer
#         encoder = LabelBinarizer()
#         newlabels = encoder.fit_transform(labels_test_temp)
#         for ii in range(len(labels_test_temp)):
#             y.append(ZeroArray(len(n_class_temp), newlabels[ii]))
#
#     y_score_temp = np.array(y_score_temp)
#     y = np.array(y)
#     # print(y, y_score)
#
#     auc = metrics.roc_auc_score(y, y_score_temp, average='weighted')
#     mavgtotal = 1
#     mae = 0
#     n_num = 0
#     for ii in range(len(n_class_temp)):
#         kkkk = len([jjjj for jjjj in range(len(y)) if np.argmax(y[jjjj]) == ii])
#         if kkkk != 0:
#             n_num += 1
#             mavgtotal = mavgtotal * len([jjjj for jjjj in range(len(y)) if np.argmax(y[jjjj]) == ii
#                                          and np.argmax(y[jjjj]) == np.argmax(y_score_temp[jjjj])]) / \
#                         len([jjjj for jjjj in range(len(y)) if np.argmax(y[jjjj]) == ii])
#         for jjj in range(len(y_score_temp)):
#             if ii == np.argmax(y[jjj]):
#                 mae += abs(1.0 - y_score_temp[jjj][ii]) / (len(n_class_temp) * len(y_score_temp))
#             else:
#                 mae += abs(y_score_temp[jjj][ii]) / (len(n_class_temp) * len(y_score_temp))
#
#     # mavg = pow(mavgtotal, 1.0/len(n_num))
#     mavg = pow(mavgtotal, 1.0 / n_num)
#     # print(acc, auc, mavg, mae, rec)
#     # return acc, auc, mavg, mae, rec
#     print(acc, auc, mavg, mae)
#     return acc, auc, mavg, mae


def measures(preds, y, y_matirx, y_sum_temp):
    # y_preds = torch.argmax(preds, 1)
    # y_preds = y_preds.cpu().numpy()
    # preds = np.array(preds)
    y_preds = np.argmax(preds, 1)
    n_class = preds.shape[1]
    # y = y.cpu().numpy()
    # print(y[0], preds[0])
    acc = metrics.accuracy_score(y, y_preds)


    # print(y_temp, preds)
    if 0 in y_sum_temp:
        auc = 0
    else:
        auc = metrics.roc_auc_score(y_matirx, preds, average='weighted')
    # auc = metrics.roc_auc_score(y_matirx, preds, average='weighted')
    # recall = metrics.recall_score(y, y_preds, average='weighted')
    mavgtotal = 1
    mae = 0
    n_num = 0
    for iii in range(n_class):
        kkk = len([jjj for jjj in range(len(y)) if y[jjj] == iii])
        if kkk != 0:
            n_num += 1
            mavgtotal = mavgtotal * len([jjj for jjj in range(len(y)) if y[jjj] == iii
                                         and y[jjj] == np.argmax(preds[jjj])]) / kkk
        for jjj in range(len(preds)):
            if iii == y[jjj]:
                mae += abs(1.0 - preds[jjj][iii]) / (n_class * len(preds))
            else:
                mae += abs(preds[jjj][iii]) / (n_class * len(preds))
    mavg = pow(mavgtotal, 1.0 / n_num)
    # return acc, auc, mavg, mae, recall
    return acc, auc, mavg, mae


def get_words_after_fs(doctrain, doctest):
    samples_temp = doctrain + doctest
    words_after_fs = get_words(samples_temp)
    wordsdf_dict = {}
    for h in range(len(words_after_fs)):
        wordsdf_dict[words_after_fs[h]] = 0
    for h in range(len(words_after_fs)):
        for m in range(len(samples_temp)):
            if words_after_fs[h] in samples_temp[m]:
                wordsdf_dict[words_after_fs[h]] += 1
    wordsdf = sorted(wordsdf_dict.items(), key=lambda d: d[1], reverse=True)
    all_words_after_fs = []
    for h in range(len(wordsdf)):
        all_words_after_fs.append(wordsdf[h][0])
    return all_words_after_fs


def doc_representation_onehot(doctrain, doctest):
    all_words = get_words_after_fs(doctrain, doctest)
    doctrainonehot = []
    doctestonehot = []
    for h in range(len(doctrain)):
        doc_temp = doctrain[h]
        vec_temp = []
        for v in all_words:
            if v in doc_temp:
                vec_temp.append(1)
            else:
                vec_temp.append(0)
        doctrainonehot.append(vec_temp)
    for h in range(len(doctest)):
        doc_temp = doctest[h]
        vec_temp = []
        for v in all_words:
            if v in doc_temp:
                vec_temp.append(1)
            else:
                vec_temp.append(0)
        doctestonehot.append(vec_temp)
    return doctrainonehot, doctestonehot

def doc_representation_tf(doctrain, doctest):
    all_words = get_words_after_fs(doctrain, doctest)
    doctraintf = []
    doctesttf = []
    for h in range(len(doctrain)):
        doc_temp = doctrain[h]
        vec_temp = []
        for v in all_words:
            if v in doc_temp:
                v_n = 0
                for w in doc_temp:
                    if w == v:
                        v_n += 1
                vec_temp.append(v_n)
            else:
                vec_temp.append(0)
        doctraintf.append(vec_temp)
    for h in range(len(doctest)):
        doc_temp = doctest[h]
        vec_temp = []
        for v in all_words:
            if v in doc_temp:
                v_n = 0
                for w in doc_temp:
                    if w == v:
                        v_n += 1
                vec_temp.append(v_n)
            else:
                vec_temp.append(0)
        doctesttf.append(vec_temp)
    return doctraintf, doctesttf


def doc_representation_tfidf(doctrain, doctest):
    # all_words = get_words_after_fs(doctrain, doctest)
    # doctraintf = []
    # doctesttf = []
    # for h in range(len(doctrain)):
    #     doc_temp = doctrain[h]
    #     vec_temp = []
    #     for v in all_words:
    #         if v in doc_temp:
    #             v_n = 0
    #             for w in doc_temp:
    #                 if w == v:
    #                     v_n += 1
    #             vec_temp.append(v_n)
    #         else:
    #             vec_temp.append(0)
    #     doctraintf.append(vec_temp)
    # for h in range(len(doctest)):
    #     doc_temp = doctest[h]
    #     vec_temp = []
    #     for v in all_words:
    #         if v in doc_temp:
    #             v_n = 0
    #             for w in doc_temp:
    #                 if w == v:
    #                     v_n += 1
    #             vec_temp.append(v_n)
    #         else:
    #             vec_temp.append(0)
    #     doctesttf.append(vec_temp)
    # idftemp = []
    # alldoc = doctrain + doctest
    # for h in range(len(all_words)):
    #     df_num = 0
    #     for ss in range(len(alldoc)):
    #         if all_words[h] in alldoc[ss]:
    #             df_num += 1
    #     idftemp.append(df_num)
    # idf = [len(alldoc)/math.log10(h) for h in idftemp]
    # doctraintfidf = []
    # doctesttfidf = []
    # for h in range(len(doctraintf)):
    #     sentencetfidf = []
    #     for v in range(len(idf)):
    #         sentencetfidf.append(doctraintf[h][v] * idf[v])
    #     doctraintfidf.append(sentencetfidf)
    # for h in range(len(doctesttf)):
    #     sentencetfidf = []
    #     for v in range(len(idf)):
    #         sentencetfidf.append(doctesttf[h][v] * idf[v])
    #     doctesttfidf.append(sentencetfidf)

    doc_train_join = [' '.join(term_list) for term_list in doctrain]
    doc_test_join = [' '.join(term_list) for term_list in doctest]
    # docall = doc_train_join + doc_test_join
    tfidf_model = TfidfVectorizer()
    # tfidf_model = TfidfVectorizer().fit(docall)
    doctraintfidf = tfidf_model.fit_transform(doc_train_join)
    # doctraintfidf = tfidf_model.transform(doc_train_join)
    doctesttfidf = tfidf_model.transform(doc_test_join)
    return doctraintfidf, doctesttfidf


def doc_representation_word_averaging(doctrain, doctest):
    print('开始加载词向量！')
    loadvecstart = time.time()
    doc_train_word_averaging = []
    doc_test_word_averaging = []
    for h in range(len(doctrain)):
        doc_temp = doctrain[h]
        if len(doc_temp) != 0:
            doc2array = np.array([ft.get_word_vector(w) for w in doc_temp])
            doc_train_word_averaging.append(np.mean(doc2array, axis=0))
        else:
            doc_train_word_averaging.append(np.zeros((embed_size), dtype=float))
    for h in range(len(doctest)):
        doc_temp = doctest[h]
        if len(doc_temp) != 0:
            doc2array = np.array([ft.get_word_vector(w) for w in doc_temp])
            doc_test_word_averaging.append(np.mean(doc2array, axis=0))
        else:
            doc_test_word_averaging.append(np.zeros((embed_size), dtype=float))
    loadvecend = time.time()
    # print(len(doc_train_word_averaging), len(doc_test_word_averaging))
    print('加载词向量的时间：', loadvecend - loadvecstart, '秒')
    return doc_train_word_averaging, doc_test_word_averaging


if __name__ == '__main__':
    # 加载数据集
    dataname = '20newsgroup'
    sample_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_small.txt'
    label_data_file = 'e:/pythonwork/newclassification/dataset/' + dataname + '_label_small.txt'
    samples = list(open(sample_data_file, "r", encoding='gb18030', errors='ignore').readlines())
    samples = [s.strip() for s in samples]
    labels = list(open(label_data_file, "r", encoding='gb18030', errors='ignore').readlines())
    labels = [s.strip() for s in labels]
    from nltk.tokenize import word_tokenize

    samples = [word_tokenize(sample) for sample in samples]

    # 获取所有词
    words = get_words(samples)
    vocab_size = len(words)
    doc_terms_list = samples

    # print(doc_terms_list)
    k = 5  # 确定 k-fold cross validation的k
    doc_train_list, labels_train_list, doc_test_list, labels_test_list = k_fold_cross_validation(doc_terms_list, labels, k)

    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('test', cell_overwrite_ok=True)
    sheet.write(0, 0, 'classifier')
    sheet.write(0, 1, 'avg_of_acc')
    sheet.write(0, 2, 'std_of_acc')
    sheet.write(0, 3, 'avg_of_auc')
    sheet.write(0, 4, 'std_of_auc')
    sheet.write(0, 5, 'avg_of_mavg')
    sheet.write(0, 6, 'std_of_mavg')
    sheet.write(0, 7, 'avg_of_mae')
    sheet.write(0, 8, 'std_of_mae')
    # sheet.write(0, 9, 'avg_of_recall')
    # sheet.write(0, 10, 'std_of_recall')
    # sheet.write(0, 11, 'avg_of_training_time')
    # sheet.write(0, 12, 'std_of_training_time')
    # sheet.write(0, 13, 'avg_of_test_time')
    # sheet.write(0, 14, 'std_of_test_time')
    sheet.write(0, 9, 'avg_of_training_time')
    sheet.write(0, 10, 'std_of_training_time')
    sheet.write(0, 11, 'avg_of_test_time')
    sheet.write(0, 12, 'std_of_test_time')
    # sheet.write(0, 13, 'No. of features')
    # sheet.write(0, 14, 'FS method')
    # row = 1
    classifier_list = ['svm_onehot', 'knn_onehot', 'lr_onehot', 'nb_onehot', 'dt_onehot', 'svm_tf', 'knn_tf', 'lr_tf',
                       'nb_tf', 'dt_tf', 'svm_tfidf', 'knn_tfidf', 'lr_tfidf', 'nb_tfidf', 'dt_tfidf', 'svm_wordavg',
                       'knn_wordavg', 'lr_wordavg',  'dt_wordavg']
    # classifier_list = ['svm_onehot', 'knn_onehot', 'lr_onehot', 'nb_onehot', 'dt_onehot', 'svm_tf', 'knn_tf', 'lr_tf',
                       # 'nb_tf', 'dt_tf']
    # classifier_list = ['svm_tfidf', 'knn_tfidf', 'lr_tfidf', 'nb_tfidf', 'dt_tfidf']
    # classifier_list = ['svm_wordavg']
    # classifier_list = ['svm', 'knn', 'lr', 'dt']
    n_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000]
    fsname_list = ['DF', 'IG', 'GI', 'DFS', 'ECE', 'CHI']
    # fsname_list = ['IG']
    embed_size = 100
    ft = load_model('E:/fasttext/cc.en.300.bin')
    # 将词向量维度减少到embed_size维
    fasttext.util.reduce_model(ft, embed_size)
    # fsname_list = ['MI', 'NEW_MI']
    # measures_total = np.zeros((k, len(classifier_list), 7))
    measures_total = np.zeros((k, len(classifier_list), 6))
    for i in range(k):
        doc_train_list_temp = doc_train_list[i]
        doc_test_list_temp = doc_test_list[i]
        labels_train = labels_train_list[i]
        labels_test = labels_test_list[i]
        encoder = preprocessing.LabelEncoder()
        labels_train = encoder.fit_transform(labels_train)
        labels_test = encoder.fit_transform(labels_test)
        doc_train = []
        doc_test = []
        for line in doc_train_list_temp:
            tmp_list = list(line.split(' '))
            doc_train.append([term for term in tmp_list])
        for line in doc_test_list_temp:
            tmp_list = list(line.split(' '))
            doc_test.append([term for term in tmp_list])
        n_class = set(labels_train)
        for j in range(len(classifier_list)):
            cc = classifier_list[j]
            classifier = 0
            if cc.startswith('svm'):
                classifier = svm.SVC(kernel="linear", probability=True)
            elif cc.startswith('knn'):
                classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
            elif cc.startswith('lr'):
                classifier = linear_model.LogisticRegression(C=1e5, max_iter=3000)
            elif cc.startswith('nb'):
                classifier = naive_bayes.MultinomialNB()
            elif cc.startswith('dt'):
                classifier = tree.DecisionTreeClassifier(max_depth=5)
            else:
                print('错误，不是正常的分类器名！！！')

            best_val_auc = 0
            best_val_mavg = 0
            best_val_mae = 0
            # best_val_recall = 0
            best_fs_num = 0
            best_fs_method = 0
            start_training = time.time()
            end_training = 0

            for fs in fsname_list:
                # (1)特征选择
                # ①得到按重要性排序的特征
                term_set_fs, runtime_fs = feature_selection(doc_train, labels_train, fs)
                # ②得到前n个特征
                new_tup_list = sorted(term_set_fs, key=lambda x: x[1], reverse=True)
                new_tup_list = np.array(new_tup_list)
                for nn in n_list:
                    term_after_fs = []
                    new_doc_train = []
                    new_doc_test = []
                    if nn <= vocab_size:
                        print('对{}数据集，第{}折，共{}折，使用{}分类器，使用{}特征选择方法，特征数量为{}'.format(dataname, i + 1, k, cc, fs, nn))
                        term_after_fs = new_tup_list[:nn, 0]
                        # print(len(term_after_fs))
                        # ③去掉训练集和测试集中的非特征词
                        for iii in range(len(doc_train)):
                            new_doc_train.append(
                                [term for term in doc_train[iii] if str(term) in term_after_fs and term != ' '])
                        for iii in range(len(doc_test)):
                            new_doc_test.append(
                                [term for term in doc_test[iii] if str(term) in term_after_fs and term != ' '])
                        # print(doc_train[0], new_doc_train[0])
                    else:
                        print('对{}数据集，第{}折，共{}折，使用{}分类器，使用{}特征选择方法，特征数量为{}'.format(dataname, i + 1,
                                                                                   k, cc, fs, vocab_size))
                        term_after_fs = new_tup_list[:vocab_size, 0]
                        # print(len(term_after_fs))
                        # ③去掉训练集和测试集中的非特征词
                        new_doc_train = doc_train
                        new_doc_test = doc_test
                    # (2)文本表示
                    # print(len(new_doc_train), len(new_doc_test))
                    # print(new_doc_train[4])
                    doc_train_representation, doc_test_representation = 0, 0
                    if cc.endswith('tfidf'):
                        doc_train_representation, doc_test_representation = doc_representation_tfidf(new_doc_train, new_doc_test)
                    elif cc.endswith('wordavg'):
                        doc_train_representation, doc_test_representation = doc_representation_word_averaging(
                            new_doc_train, new_doc_test)
                    elif cc.endswith('onehot'):
                        doc_train_representation, doc_test_representation = doc_representation_onehot(
                            new_doc_train, new_doc_test)
                    elif cc.endswith('tf'):
                        doc_train_representation, doc_test_representation = doc_representation_tf(
                            new_doc_train, new_doc_test)
                    # (3)分类
                    # index = 0
                    # for temp in doc_train_representation:
                    #     if len(temp) == 100:
                    #         print(temp, index)
                    #     index += 1
                    classifier.fit(doc_train_representation, labels_train)
                    # predict the labels on validation dataset
                    predictions = classifier.predict(doc_test_representation)
                    y_score = classifier.predict_proba(doc_test_representation)
                    y_class = []
                    for iii in range(len(labels_test)):
                        if labels_test[iii] not in y_class:
                            y_class.append(labels_test[iii])
                    y_temp = np.zeros(y_score.shape).tolist()
                    for ii in range(len(labels_test)):
                        y_temp[ii][labels_test[ii]] = 1
                    y_sum = np.sum(y_temp, axis=0).tolist()
                    print(y_sum)
                    # accuracy, auc, mavg, mae, recall = measures(predictions, y_score, labels_test, n_class)
                    accuracy, auc, mavg, mae = measures(y_score, labels_test, y_temp, y_sum)
                    print('>>test:', accuracy, auc, mavg, mae)
                    if accuracy > best_val_acc:
                        file_list = os.listdir('model/')
                        for file in file_list:
                            if file.startswith('{data}_best_{kth}_{classifiername}'.format(data=dataname, kth=i, classifiername=cc)):
                                os.remove('model/' + file)
                                # 保存模型
                        with open('model/{data}_best_{kth}_{classifiername}_{fnum}_{fsname}'.format(data=dataname,kth=i,
                                                                                             classifiername=cc,
                                                                                             fnum=nn,
                                                                                             fsname=fs),
                                  'wb') as f:
                            pickle.dump(classifier, f)
                            f.close()
                        # 保存特征
                        with open(
                                'model/{data}_best_{kth}_{classifiername}_{fnum}_{fsname}_features.txt'.format(
                                    data=dataname, kth=i, classifiername=cc,
                                    fnum=nn,
                                    fsname=fs),
                                'w', encoding='utf-8') as f:
                            for word in term_after_fs:
                                f.write(word + '\n')
                            f.close()
                        best_val_acc = accuracy
                        best_val_auc = auc
                        best_val_mavg = mavg
                        best_val_mae = mae
                        # best_val_recall = recall
                        best_fs_num = nn
                        best_fs_method = fs
                        end_training = time.time()
                        if nn > vocab_size:
                            break


            # 加载模型
            pickle_in = open('model/{data}_best_{kth}_{classifiername}_{fnum}_{fsname}'.format(data=dataname, kth=i,
                                                                                        classifiername=cc,
                                                                                        fnum=best_fs_num,
                                                                                        fsname=best_fs_method), 'rb')
            clf = pickle.load(pickle_in)
            features = []
            # 加载特征
            with open('model/{data}_best_{kth}_{classifiername}_{fnum}_{fsname}_features.txt'.format(data=dataname, kth=i,
                                                                                              classifiername=cc,
                                                                                              fnum=best_fs_num,
                                                                                              fsname=best_fs_method),
                      'r', encoding='utf-8') as f:
                for word in f.readlines():
                    features.append(word.replace('\n', ''))
                f.close()
            # 去掉训练集和测试集中的非特征词
            new_doc_train = []
            new_doc_test = []
            for iii in range(len(doc_train)):
                new_doc_train.append(
                    [term for term in doc_train[iii] if str(term) in features and term != ' '])
            for iii in range(len(doc_test)):
                new_doc_test.append(
                    [term for term in doc_test[iii] if str(term) in features and term != ' '])
            # print(doc_train[0], new_doc_train[0])
            # 文本表示

            doc_train_representation, doc_test_representation = 0, 0
            if cc.endswith('tfidf'):
                doc_train_representation, doc_test_representation = doc_representation_tfidf(new_doc_train,
                                                                                                  new_doc_test)
            elif cc.endswith('wordavg'):
                doc_train_representation, doc_test_representation = doc_representation_word_averaging(
                    new_doc_train, new_doc_test)
            elif cc.endswith('onehot'):
                doc_train_representation, doc_test_representation = doc_representation_onehot(
                    new_doc_train, new_doc_test)
            elif cc.endswith('tf'):
                doc_train_representation, doc_test_representation = doc_representation_tf(
                    new_doc_train, new_doc_test)
            # 分类
            # classifier.fit(doc_train_representation, labels_train)
            # predict the labels on validation dataset
            # print(len(doc_train_representation[0]), len(doc_test_representation[0]))
            predictions = clf.predict(doc_test_representation)
            y_score = clf.predict_proba(doc_test_representation)
            y_class = []
            for iii in range(len(labels_test)):
                if labels_test[iii] not in y_class:
                    y_class.append(labels_test[iii])
            y_temp = np.zeros(y_score.shape).tolist()
            for ii in range(len(labels_test)):
                y_temp[ii][labels_test[ii]] = 1
            y_sum = np.sum(y_temp, axis=0).tolist()
            # measures_total[i][j][0], measures_total[i][j][1], measures_total[i][j][2], measures_total[i][j][3], measures_total[i][j][4] = measures(predictions, y_score, labels_test, n_class)
            measures_total[i][j][0], measures_total[i][j][1], measures_total[i][j][2], measures_total[i][j][3] = measures(y_score, labels_test, y_temp, y_sum)
            end_test = time.time()
            training_time = end_training - start_training
            test_time = end_test - end_training
            measures_total[i][j][4] = training_time
            measures_total[i][j][5] = test_time
            print('特征选择后的最优表现：', measures_total[i][j][0], measures_total[i][j][1], measures_total[i][j][2], measures_total[i][j][3], measures_total[i][j][4], measures_total[i][j][5])

            print('=================================')
    measures_total_avg = np.mean(measures_total, axis=0)
    measures_total_std = np.std(measures_total, axis=0)
    for i in range(len(measures_total_avg)):
        sheet.write(i + 1, 0, classifier_list[i])
        sheet.write(i + 1, 1, measures_total_avg[i][0])
        sheet.write(i + 1, 2, measures_total_std[i][0])
        sheet.write(i + 1, 3, measures_total_avg[i][1])
        sheet.write(i + 1, 4, measures_total_std[i][1])
        sheet.write(i + 1, 5, measures_total_avg[i][2])
        sheet.write(i + 1, 6, measures_total_std[i][2])
        sheet.write(i + 1, 7, measures_total_avg[i][3])
        sheet.write(i + 1, 8, measures_total_std[i][3])
        # sheet.write(i + 1, 9, measures_total_avg[i][4])
        # sheet.write(i + 1, 10, measures_total_std[i][4])
        # sheet.write(i + 1, 11, measures_total_avg[i][5])
        # sheet.write(i + 1, 12, measures_total_std[i][5])
        # sheet.write(i + 1, 13, measures_total_avg[i][6])
        # sheet.write(i + 1, 14, measures_total_std[i][6])
        sheet.write(i + 1, 9, measures_total_avg[i][4])
        sheet.write(i + 1, 10, measures_total_std[i][4])
        sheet.write(i + 1, 11, measures_total_avg[i][5])
        sheet.write(i + 1, 12, measures_total_std[i][5])

    book.save('E:/成都理工大学重要文件夹/结果/' + dataname + '_criteria.xls')
