import csv

with open("dataset/ptbxl_database.csv", 'r') as raw_ptbxl_fp:
    with open("dataset/scp_statements.csv", 'r') as raw_stmts_fp:
        raw_ptbxl_csv = csv.DictReader(raw_ptbxl_fp)
        raw_stmts_csv = csv.DictReader(raw_stmts_fp)

        lst_ptbxl = []
        lst_stmts = []

        for i in raw_ptbxl_csv:
            lst_ptbxl.append(i)

        for i in raw_stmts_csv:
            lst_stmts.append(i)

        stmt_dict = dict()
        lst_out = []

        for i in lst_stmts:
            stmt_dict[i['']] = i['diagnostic_class']

        for i in lst_ptbxl:
            if i['scp_codes'].count('NORM'):
                lst_out.append([0, i['filename_lr'].split('.')[0], i['strat_fold']])
            else:
                lst_out.append([1, i['filename_lr'].split('.')[0], i['strat_fold']])

        header = ['class', 'fname', 'fold']

        with open('dataset/label.csv', 'w') as outfp:
            writer = csv.writer(outfp)
            writer.writerow(header)
            writer.writerows(lst_out)