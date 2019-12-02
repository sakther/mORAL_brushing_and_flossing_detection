import pandas as pd
import numpy as np

def save_as_csv(X, Y, feature_name, output_dir, output_filename='features_and_labels.csv'):
    # print(X[0], len(X[0]), len(feature_name))
    data = np.array(X)
    pd_data = pd.DataFrame(data=data,columns=feature_name)
    pd_data['label'] = Y

    pd_data.to_csv(output_dir + output_filename, encoding='utf-8', index=False)
    print('---feature saved in ', output_dir + output_filename)

def append_to_file(filename, txt):
    fh = open(filename, 'a')
    fh.write(txt + '\n')
    fh.close()

