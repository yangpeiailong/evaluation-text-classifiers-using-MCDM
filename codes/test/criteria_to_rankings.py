import xlrd
import numpy as np
import pandas as pd
import xlwt
from tqdm import tqdm
import os
path = 'D:/成都理工大学重要文件夹/Evaluation of Text Classification Methods in Small Samples by AHP-PROMETHEE/Attachment/friedman_test/criteria_across_datasets'
file_name_list = []
path_list = []
b_or_c = ['b', 'b', 'c', 'b', 'c', 'c', 'c', 'c', 'c', 'c']
for file_name in os.listdir(path):
    file_name_list.append(file_name)
    # print(path + '/' + file_name)
    path_list.append(path + '/' + file_name)

for ii in tqdm(range(len(path_list))):
    classifiers = []
    book = xlrd.open_workbook(path_list[ii])
    table = book.sheet_by_index(0)
    data = []
    nrows = table.nrows  # 行数
    ncols = table.ncols # 列数
    # print(ncols)
    for i in range(1, nrows):
        classifiers.append(table.cell(i, 0).value)
    # print(classifiers)

    datanamelist = [table.cell(0,j).value for j in range(1, ncols)]
    book1 = xlwt.Workbook(encoding='utf-8')
    sheet = book1.add_sheet('Sheet1', cell_overwrite_ok=True)
    sheet.write(0, 0, 'classifier')
    for i in range(len(classifiers)):
        sheet.write(i + 1, 0, classifiers[i])
    for i in range(len(datanamelist)):
        sheet.write(0, i + 1, datanamelist[i])

    for i in range(1, ncols):
        print(i)
        data = np.array([table.cell(j, i).value for j in range(1, nrows)])
        # print(data)
        ser = pd.Series(data)
        if b_or_c[ii] == 'c':
            ser = ser.rank(ascending=True) # True为升序，False为降序
        else:
            ser = ser.rank(ascending=False)  # True为升序，False为降序
        ser_list = ser.tolist()
        for j in range(len(ser_list)):
            sheet.write(j + 1, i, ser_list[j])

    book1.save('D:/成都理工大学重要文件夹/Evaluation of Text Classification Methods in Small Samples by AHP-PROMETHEE/Attachment/friedman_test/criteria_across_datasets_to_rankings/' + file_name_list[ii])




