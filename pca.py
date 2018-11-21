import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#===========================
#Part 1
#Singular Value Decompostion to investigate eigenvalues and eigenvectors of observation matrix
#===========================

##Manual construction of observations, X_obs, in 2 dimensional space:
#observations characterised by features X1 and X2
X1 = np.array([-0.066,-0.124, 0.259, 0.289, -0.318, -0.015, -0.060, 0.140, 0.203, 0.249, -0.229, 0.174, 0.149, -0.343, -0.308])
X2 = np.array([-0.204, -0.079, 0.565, 0.607, -0.425,-0.135, 0.011, 0.411, 0.350, 0.535, -0.668, 0.128, 0.317, -0.543, -0.870])

X_obs = np.array([X1,X2])
plt.scatter(X1, X2)
plt.ylabel("X2")
plt.xlabel("X1")
plt.title("Observed data")
plt.show()

#Performing a Singular Value Decomposition on the Variance Covariance Matrix:
VarCov = X_obs @ X_obs.T
u,s,v = np.linalg.svd(VarCov) 

#Performing a Singular Value Decomposition on the observation Matrix:
u2,s2,v2 = np.linalg.svd(X_obs)

#eigenvectors that make up the data are the same as those that capture the variance in the data
print("Eigenvectors VarCov matrix:")
print(v)
print("")
print("Eigenvectors X_obs matrix:")
print(u2)
print("")

print("Eigenvalues of VarCov matrix:")
print(s)
print("")
print("Eigenvalues/Squared Singular values of X_obs matrix:")
print(s2**2)
print("")
#s = s2**2 -> eigenvalues that indicate the relative importance of each eigenvector that capture the variance in the data


#===========================
#Part 2
#Dimensionality reduction using PCA
#===========================

df = pd.DataFrame({"X1":X1, "X2":X2})

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(df)

print("X_reduced shape:")
print(X_reduced.shape)
print("")

print("The principal component: same as eigenvector with largest eigenvalue as seen above ")
print(pca.components_)
print("")

# one dimensional representation of data:
plt.scatter(x = X_reduced, y =np.zeros(len(X_reduced)))
plt.title("One dimensional representation along real line")
plt.xlabel("first principle component")
plt.show()

#projecting back into 2 dimensional space
x1_approx = (X_reduced.reshape(1,15)[0])*(pca.components_.reshape(1,2)[0][0])
x2_approx = (X_reduced.reshape(1,15)[0])*(pca.components_.reshape(1,2)[0][1])
projected_data = pd.DataFrame({"X1_approx":x1_approx, "X2_approx":x2_approx})

plt.scatter(X1, X2, alpha = 0.5, label = "original")
plt.ylabel("X2")
plt.xlabel("X1")
plt.title("Projecting reduced data back into 2 dimensional space")
plt.scatter(x1_approx,x2_approx, alpha =0.5,label = "approx using pca")
plt.legend()
plt.show()

print("Proportion of the variance in the data that is captured by the first principle component")
print(pca.explained_variance_ratio_)
