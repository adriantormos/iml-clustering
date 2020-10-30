from scipy.io import arff
import numpy as np
import pandas as pd

np.random.seed(1)

data, meta = arff.loadarff('breast-w.arff')
df = pd.DataFrame(data)
print(df)
print(np.shape(df))

df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()# Remove NaN elements
class_name = df.columns[-1]# The last column is considered the class

print('df with no Nan=', np.shape(df))


classes = meta[class_name][1]
y = np.array([classes.index(e.decode('utf-8')) for e in data[class_name]])

type_list = np.array(meta.types())
print(type_list)

nominal_bool = (type_list == 'numeric')
nominal_columns = np.array(meta.names())[nominal_bool]

# shape=(n_samples, n_features)
X = df[nominal_columns].as_matrix()
print('X=', X)

# Scaling
mean_vec = np.matrix(np.mean(X, axis=0))
n, m = X.shape
M = np.repeat(mean_vec, n, axis=0)
M = np.array(M)
Xc = (X - M) # Xc = X centered
#sd = 1
sd = np.std(Xc, axis=0)
Xcs = Xc/sd # Xcs = X centered and scaled

# Problem with division by 0
Xcs = np.nan_to_num(Xcs)
print(Xcs
      )
#np.savetxt('breastProcessed', df)


