import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
#データベースをダウンロードする
iris_data = load_iris()
#データベースのデータを読み取る
X = iris_data['data']
y = iris_data['target']
feature_names = iris_data['feature_names']
target_names = iris_data['target_names']

X_scaled = StandardScaler().fit_transform(X)
#PCA関数を実例化する
pca = PCA()
X_trans = pca.fit_transform(X_scaled)

print(pd.DataFrame(pca.components_, columns=feature_names))
print(f'特徴1の寄与度{pca.explained_variance_ratio_[0]}')
print(f'特徴2の寄与度{pca.explained_variance_ratio_[1]}')
print(f'特徴3の寄与度{pca.explained_variance_ratio_[2]}')
print(f'特徴4の寄与度{pca.explained_variance_ratio_[3]}')
#寄与度を表現する
fig, ax = plt.subplots()


#写像した 2 次元平面を表現する
X0 = X_trans[y == 0]
X1 = X_trans[y == 1]
X2 = X_trans[y == 2]


ax.scatter(X0[:, 0], X0[:, 1])
ax.scatter(X1[:, 0], X1[:, 1])
ax.scatter(X2[:, 0], X2[:, 1])

ax.set_xlabel("Component-1")
ax.set_ylabel("Component-2")

plt.show()