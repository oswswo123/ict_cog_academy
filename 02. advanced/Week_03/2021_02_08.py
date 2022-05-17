#%% 머신러닝 연습

from sklearn.datasets import load_iris, load_digits
import pandas as pd

digit = load_digits()
iris = load_iris()

iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)