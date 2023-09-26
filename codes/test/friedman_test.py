import xlrd
from scipy.stats import friedmanchisquare
import os
from tqdm import tqdm

# path = 'D:/成都理工大学重要文件夹/Evaluation of Text Classification Methods in Small Samples by AHP-PROMETHEE/Attachment/friedman_test/criteria_across_datasets_to_rankings'
# file_name_list = []
# path_list = []
# for file_name in os.listdir(path):
#     file_name_list.append(file_name)
#     # print(path + '/' + file_name)
#     path_list.append(path + '/' + file_name)

path_list = ['D:/成都理工大学重要文件夹/Evaluation of Text Classification Methods in Small Samples by AHP-PROMETHEE/Attachment/simpler is better/bert_total_promethee_rankings.xls']
dict1 = {}
for ii in tqdm(range(len(path_list))):
    classifiers = []
    book = xlrd.open_workbook(path_list[ii])
    table = book.sheet_by_index(0)
    data = []
    nrows = table.nrows  # 行数
    ncols = table.ncols # 列数
    for i in range(1, nrows):
        classifiers.append(table.cell(i, 0).value)

    rankings_for_all_classifiers = []
    for i in range(1, nrows):
        rankings_for_each_classifier = []
        for j in range(1, ncols):
            rankings_for_each_classifier.append(float(table.cell(i,j).value))
        rankings_for_all_classifiers.append(rankings_for_each_classifier)

    # print(rankings_for_all_classifiers)
    # stat, p = friedmanchisquare(rankings_for_all_classifiers[0],rankings_for_all_classifiers[1], rankings_for_all_classifiers[2],
    #                             rankings_for_all_classifiers[3],rankings_for_all_classifiers[4], rankings_for_all_classifiers[5],
    #                             rankings_for_all_classifiers[6],rankings_for_all_classifiers[7], rankings_for_all_classifiers[8],
    #                             rankings_for_all_classifiers[9], rankings_for_all_classifiers[10],
    #                             rankings_for_all_classifiers[11],
    #                             rankings_for_all_classifiers[12], rankings_for_all_classifiers[13],
    #                             rankings_for_all_classifiers[14],
    #                             rankings_for_all_classifiers[15], rankings_for_all_classifiers[16],
    #                             rankings_for_all_classifiers[17],
    #                             rankings_for_all_classifiers[18], rankings_for_all_classifiers[19],
    #                             rankings_for_all_classifiers[20],
    #                             rankings_for_all_classifiers[21], rankings_for_all_classifiers[22],
    #                             rankings_for_all_classifiers[23],
    #                             rankings_for_all_classifiers[24], rankings_for_all_classifiers[25],
    #                             rankings_for_all_classifiers[26],
    #                             rankings_for_all_classifiers[27], rankings_for_all_classifiers[28],
    #                             rankings_for_all_classifiers[29],
    #                             rankings_for_all_classifiers[30], rankings_for_all_classifiers[31],
    #                             rankings_for_all_classifiers[32],
    #                             rankings_for_all_classifiers[33], rankings_for_all_classifiers[34]
    #                             )
    stat, p = friedmanchisquare(rankings_for_all_classifiers[0], rankings_for_all_classifiers[1],
                                rankings_for_all_classifiers[2],
                                rankings_for_all_classifiers[3], rankings_for_all_classifiers[4],
                                rankings_for_all_classifiers[5],
                                rankings_for_all_classifiers[6], rankings_for_all_classifiers[7],
                                rankings_for_all_classifiers[8],
                                rankings_for_all_classifiers[9], rankings_for_all_classifiers[10],
                                rankings_for_all_classifiers[11])
    # stat, p = friedmanchisquare(rankings_for_all_classifiers[0], rankings_for_all_classifiers[1],
    #                             rankings_for_all_classifiers[2],
    #                                                         rankings_for_all_classifiers[3],rankings_for_all_classifiers[4], rankings_for_all_classifiers[5],
    #                                                         rankings_for_all_classifiers[6],rankings_for_all_classifiers[7], rankings_for_all_classifiers[8],
    #                                                         rankings_for_all_classifiers[9], rankings_for_all_classifiers[10],
    #                                                         rankings_for_all_classifiers[11],
    #                                                         rankings_for_all_classifiers[12], rankings_for_all_classifiers[13],
    #                                                         rankings_for_all_classifiers[14],
    #                                                         rankings_for_all_classifiers[15], rankings_for_all_classifiers[16],
    #                                                         rankings_for_all_classifiers[17],
    #                                                         rankings_for_all_classifiers[18], rankings_for_all_classifiers[19])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('不能拒绝原假设，样本集分布相同')
    else:
        print('拒绝原假设，样本集分布可能不同')
    # dict1[file_name_list[ii]] = p


# print(dict1)