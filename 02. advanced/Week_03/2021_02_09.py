#%% scikit-learn learning curve

from sklearn.datasets import load_iris

data = load_iris()

from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier

result = learning_curve(KNeighborsClassifier(), data.data, data.target)

#%% fashion mnist

import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# EDA (Exploratory Data Analysis)
plt.imshow(X_train[0], cmap='gray')
plt.colorbar()
plt.grid(False)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)



# import mglearn
# mglearn.plot_cross_validation.plot_cross_validation()

X_train_reshape = X_train.reshape(60000, -1)
X_test_reshape = X_test.reshape(10000, -1)

# tensorflow에서 load한 data를 scikit learn의 KNN으로 train
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_reshape, y_train)

# KNN이라 너무 오래걸림
# print(knn.score(X_test_reshape, y_test))

# 정확한 비교를 위한 객관적 기준, 평가
# - 데이터가 어느정도 충분하다고 판단 : holdout
# - 데이터의 양이 적다고 판단 : cross validation
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

# 데이터가 너무 많아 오래걸리니 25개만 5등분으로 평가
# cross validation의 가장 유명한 3가지
print(cross_val_score(KNeighborsClassifier(), X_train_reshape[:25], y_train[:25], cv=5))
print()
print(cross_val_predict(KNeighborsClassifier(), X_train_reshape[:25], y_train[:25], cv=5))
print()
print(cross_validate(KNeighborsClassifier(), X_train_reshape[:25], y_train[:25], cv=5))


