import xlrd
from scipy.stats import wilcoxon

path = 'D:/成都理工大学重要文件夹/Evaluation of Text Classification Methods in Small Samples by AHP-PROMETHEE/Attachment/simpler is better/dl_total_promethee_scores.xls'
classifiers = []
book = xlrd.open_workbook(path)
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
print(wilcoxon(rankings_for_all_classifiers[18], rankings_for_all_classifiers[19]))