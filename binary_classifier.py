from classifiers.binary import BinaryModel

# load the data
import pandas as pd

df = pd.read_csv('./classifiers/data/counterfit.csv')

bm = BinaryModel()

bm.train_test(df)

if __name__ == '__main__':
    print(bm.predict(df))




