import xlrd
import numpy as np
import math
import xlwt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm

def pf_linear(diff, s1, s2):
    p = 0
    if s1 < diff <= s2:
        p = (diff - s1) / (s2 - s1)
    elif diff > s2:
        p = 1
    return p

def promethee(data, w, borc):
    data_new = np.zeros((data.shape[0], data.shape[1]), np.float32)
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            if borc[jj] == 'b':
                data_new[ii][jj] = data[ii][jj]
            elif borc[jj] == 'c':
                data_new[ii][jj] = np.max(data[:, jj]) - data[ii][jj]
    philist = np.zeros(data.shape[0], np.float32)
    for ii in range(data.shape[0]):
        phip = 0
        phim = 0
        for jj in range(data.shape[0]):
            if jj != ii:
                paip = 0
                paim = 0
                for kk in range(data.shape[1]):
                    difflist = []
                    for iiii in range(data.shape[0]):
                        for jjjj in range(data.shape[0]):
                            if iiii != jjjj and data_new[iiii][kk] - data_new[jjjj][kk] >= 0:
                                difflist.append(data_new[iiii][kk] - data_new[jjjj][kk])
                    # print(kk)
                    # print(data[:, kk])
                    # print(difflist)
                    # print(length1, length2)
                    aaa = np.zeros(data.shape[0], np.float32)
                    bbb = np.zeros(data.shape[0], np.float32)
                    if 0 <= kk <= 6:
                        diff = data_new[ii][kk] - data_new[jj][kk]
                        p = pf_linear(diff, 0.01, 0.1)
                        paip += p * w[kk]
                    elif 7 <= kk <= 8:
                        amax = np.max(difflist)
                        amin = np.min(difflist)
                        diff = data_new[ii][kk] - data_new[jj][kk]
                        p = pf_linear(diff, amin, amax)
                        paip += p * w[kk]
                phip += 1 / (data.shape[0] - 1) * paip
                phim += 1 / (data.shape[0] - 1) * paim
        philist[ii] = phip - phim
    sorted_score_index = np.argsort(philist)[:: -1]
    rset_temp = philist[sorted_score_index]
    rset_temp = rset_temp.tolist()
    new_sort = []
    for ii in range(len(philist)):
        new_sort.append(rset_temp.index(philist[ii]) + 1)
    return philist, new_sort


# datanamelist = ['20newsgroup', 'amazon_cells', 'cade', 'farm', 'Pang&Lee', 'reuter8', 'sentence']

datanamelist = ['20newsgroup', 'amazon_review_full_csv (5)', 'bbcsport', 'dbpedia',
                'reuter8', 'WOS5736', 'sentence', 'yelp_review_full_csv (5)', 'ag_news' ,'amazon_cells',
                'amazon_review_polarity_csv (2)', 'farm', 'Pang&Lee', 'yelp_review_polarity_csv (2)', 'IMDB']
w = [0.34606393, 0.18103399, 0.137613, 0.0778131, 0.137613,  0.03401007, 0.02005291, 0.02595, 0.02595]
# w分别为avg_of_acc	std_of_acc	avg_of_auc	std_of_auc	avg_of_mavg	std_of_mavg	avg_of_mae	std_of_mae	avg_of_training_time	std_of_training_time	avg_of_test_time	std_of_test_time
# 与文章顺序不同
borc = ['b', 'c','b', 'c', 'b', 'b', 'c', 'c', 'c', ]

classifiers = []
book = xlrd.open_workbook(r"D:/pythonwork/newclassification/results3/%s_bert_total_criteria.xls" % datanamelist[0])
table = book.sheet_by_index(0)
data = []
nrows = table.nrows  # 行数
for i in range(1, nrows):
    classifiers.append(table.cell(i, 0).value)
# print(classifiers)

book1 = xlwt.Workbook(encoding='utf-8')
book2 = xlwt.Workbook(encoding='utf-8')
sheet = book1.add_sheet('test', cell_overwrite_ok=True)
sheet2 = book2.add_sheet('test', cell_overwrite_ok=True)
sheet.write(0, 0, 'classifier')
sheet2.write(0, 0, 'classifier')
for i in range(len(classifiers)):
    sheet.write(i + 1, 0, classifiers[i])
    sheet2.write(i + 1, 0, classifiers[i])
for i in range(len(datanamelist)):
    sheet.write(0, i + 1, datanamelist[i])
    sheet2.write(0, i + 1, datanamelist[i])

for i in tqdm(range(len(datanamelist)), ncols=100):
    book = xlrd.open_workbook(r"D:/pythonwork/newclassification/results3/%s_bert_total_criteria.xls" % datanamelist[i])
    table = book.sheet_by_index(0)
    data = []
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    for k in range(1, nrows):
        temp = []
        for l in range(1, ncols):
            temp.append(table.cell(k, l).value)
        data.append(temp)
    data = np.array(data)
    promethee_score, promethee_rank = promethee(data, w, borc)
    for j in range(len(classifiers)):
        sheet.write(j+1, i+1, promethee_rank[j])
        sheet2.write(j+1, i+1, float(promethee_score[j]))
# book1.save('D:/pythonwork/newclassification/results3/dl_total_promethee_rankings.xls')
book2.save('D:/pythonwork/newclassification/results3/bert_total_promethee_scores.xls')