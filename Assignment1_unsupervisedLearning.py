#!/usr/bin/env python
# coding: utf-8

# In[11]:


#PART - A


# In[1]:


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


# In[2]:


#Retrieve and load the mnist_784
mnist = fetch_openml('mnist_784', version=1, as_frame=False)


# In[3]:


#Display each digit.
data = mnist.data.astype(np.uint8)
labels = mnist.target.astype(np.uint8)
print(labels)
plt.figure(figsize=(10, 10))
for i in range(10):
    digit_indices = np.where(labels == i)[0]
    for j in range(10):
        plt.subplot(10, 10, i * 10 + j + 1)
        plt.imshow(data[digit_indices[j]].reshape(28, 28), cmap='gray')
        plt.title(f"Digit: {i}")
        plt.axis('off')

plt.tight_layout()
plt.show()


# In[4]:


#Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio.
n_components = 2

pca = PCA(n_components=n_components)

pca.fit(data)

#The below will find the 1st and 2nd component 
first_principal_component = pca.components_[0]
second_principal_component = pca.components_[1]

#This will find the explained variance ratio from the above pca
explained_variance_ratio = pca.explained_variance_ratio_

print(f"Explained Variance Ratio for 1st Principal Component: {explained_variance_ratio[0]}")
print(f"Explained Variance Ratio for 2nd Principal Component: {explained_variance_ratio[1]}")


# In[5]:


#Plot the projections of the 1st and 2nd principal component onto a 1D hyperplane.
projection_1st_component = np.dot(data, first_principal_component)
projection_2nd_component = np.dot(data, second_principal_component)

#The below will plot the hyperplane for 1st principal component
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(projection_1st_component, np.zeros_like(projection_1st_component), c=labels, cmap='viridis', marker='o')
plt.title("Projection onto 1st Principal Component")
plt.xlabel("Projection Value")
plt.yticks([])

#The below will plot the hyperplane for 2nd principal component
plt.subplot(1, 2, 2)
plt.scatter(projection_2nd_component, np.zeros_like(projection_2nd_component), c=labels, cmap='viridis', marker='o')
plt.title("Projection onto 2nd Principal Component")
plt.xlabel("Projection Value")
plt.yticks([])

plt.tight_layout()
plt.show()


# In[6]:


#Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions.
n_components = 154

ipca = IncrementalPCA(n_components=n_components)

batch_size = 2000
for i in range(0, data.shape[0], batch_size):
    batch = data[i:i+batch_size]
    ipca.partial_fit(batch)

mnist_reduced = ipca.transform(data)


# In[7]:


#Display the original and compressed digits 
sample_indices = np.random.choice(mnist_reduced.shape[0], 10, replace=False)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 1)
    plt.imshow(ipca.inverse_transform(mnist_reduced[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')

plt.subplot(1, 2, 2)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 11)
    plt.imshow(ipca.inverse_transform(mnist_reduced[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Compressed')
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[8]:


#PART - B


# In[47]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[51]:


#Generate Swiss roll dataset
n_samples = 1000
X, color = make_swiss_roll(n_samples, noise=0.2, random_state=42)


# In[52]:


#Plot the resulting generated Swiss roll dataset.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll Dataset")
plt.show()


# In[55]:


#Use Kernel PCA (kPCA) with linear kernel, a RBF kernel, and a sigmoid kernel.
kpca_linear = KernelPCA(kernel="linear", n_components=2)
kpca_rbf = KernelPCA(kernel="rbf", gamma=0.04, n_components=2)
kpca_sigmoid = KernelPCA(kernel="sigmoid", gamma=0.001, n_components=2)

X_linear = kpca_linear.fit_transform(X)
X_rbf = kpca_rbf.fit_transform(X)
X_sigmoid = kpca_sigmoid.fit_transform(X)

plt.figure(figsize=(15, 5))

#The below will plot the linear kernel graph
plt.subplot(131)
plt.scatter(X_linear[:, 0], X_linear[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Linear Kernel")

#The below will plot the RBF kernel graph
plt.subplot(132)
plt.scatter(X_rbf[:, 0], X_rbf[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with RBF Kernel")

#The below will plot the Sigmoid kernel graph
plt.subplot(133)
plt.scatter(X_sigmoid[:, 0], X_sigmoid[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Sigmoid Kernel")

plt.tight_layout()
plt.show()


# In[60]:


#Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
#Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at 
#the end of the pipeline. Print out best parameters found by GridSearchCV.
n_samples = 1000
X, color = make_swiss_roll(n_samples, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, color, test_size=0.2, random_state=42)

threshold = np.median(color)  # You can adjust the threshold as needed
y_train_binary = (y_train > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the data
    ('kpca', KernelPCA(kernel='rbf')),  # Default kernel choice (you can change this)
    ('classifier', LogisticRegression(max_iter=1000))
])

param_grid = {
    'kpca__kernel': ['rbf', 'sigmoid', 'poly'],  # Kernels to search
    'kpca__gamma': [0.001, 0.01, 0.1, 1.0, 10.0]  # Gamma values to search
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train_binary)

#The below will print the best parameters found from the grid search
best_params = grid_search.best_params_
print("Best Parameters:")
print(best_params)

#The below will print the best classifier from the grid search
best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test)

#The below will print the accuracy score
accuracy = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy on Test Set: {accuracy:.2f}")


# In[61]:


#Plot the results from using GridSearchCV
scores = grid_search.cv_results_["mean_test_score"]
gammas = [params["kpca__gamma"] for params in grid_search.cv_results_["params"]]
plt.figure(figsize=(10, 6))
plt.scatter(gammas, scores, c=scores, cmap=plt.cm.viridis)
plt.colorbar(label="Mean Test Score")
plt.xlabel("Gamma")
plt.ylabel("Mean Test Score")
plt.title("GridSearchCV Results")
plt.show()


# In[ ]:




