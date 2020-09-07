"""
@author: aswamy
@github: hsakas
"""

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class BinaryModel:
    def __init__(self):
        self.lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')

    def train_test(self, dataframe: DataFrame) -> None:
        x, y = dataframe.iloc[:, 1:-1], dataframe.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        self.lr.fit(x_train, y_train)
        self.lr.predict(x_test)
        print(f'Score on test data -> {round(self.lr.score(x_test, y_test), 4)}')

    def predict(self, dataframe: DataFrame) -> DataFrame:
        x = dataframe.iloc[:, 1:-1]
        return self.lr.predict(x)