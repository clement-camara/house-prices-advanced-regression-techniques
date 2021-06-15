#!/usr/bin/env python
# coding: utf-8




import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

from sklearn.metrics import mean_squared_error

from sklearn import set_config
set_config(display='diagram')

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import time


st.title('House prices: \n Machine learning with sklearn')

from PIL import Image
image = Image.open('house_image.jpeg')
st.image(image, caption='House')


st.title('Data')



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



train.drop(['Id'], axis=1, inplace=True)


print(' train :', train.shape)
print(' test :', test.shape)



st.write(train)



# ## Reflexion
st.title('Reflexion')


# prix moyen d'un appartement 180921 dollars
#train.SalePrice.describe()

st.title('distribution')

# ## distribution



fig, ax = plt.subplots()
# si c'est asymétrique c'est bon si non je convertie en logarithmique
print("coefficient d'asymétrie: ", train.SalePrice.skew())
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)
plt.hist(train.SalePrice,color='blue')
fig.suptitle("Visualisation de l'asymétrie de ma target")
plt.show() #Positive asymmetry or to the right.
st.pyplot(fig)




fig, ax = plt.subplots()
# ma target est log c'est bon
target=np.log(train.SalePrice)
print("La valeur de y est : ", target.skew())
plt.hist(target,color="blue")
fig.suptitle("Visualisation avec la target logarithmiqué")
plt.show()
st.pyplot(fig)


st.title('Outliers')
# ## Outliers


image2 = Image.open('GarageArea.jpeg')
st.image(image2, caption='Outliers 1')

image3 = Image.open('GrLivArea.jpeg')
st.image(image3, caption='Outliers 2')


fig = plt.scatter(x=train.GarageArea,y=np.log(train.SalePrice)) # je recrée la target
plt.show()


plt.scatter(x=train.GrLivArea,y=np.log(train.SalePrice)) # je recrée la target
plt.show()



st.code("""
train=train[train.GarageArea <1200]
train=train[train.GrLivArea <4000]
""", language="python")

train=train[train.GarageArea <1200]
train=train[train.GrLivArea <4000]


st.title('feature engenering')

st.code("""
# Ajouter les zones d'habitation et la zone du sous-sol pour créer une nouvelle fonctionnalité TotArea
test["TotArea"] = test["GrLivArea"] + test["TotalBsmtSF"]
train["TotArea"] = train["GrLivArea"] + train["TotalBsmtSF"]
""", language="python")

test["TotArea"] = test["GrLivArea"] + test["TotalBsmtSF"]
train["TotArea"] = train["GrLivArea"] + train["TotalBsmtSF"]

st.code("""
drop_col_test = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', '3SsnPorch']
test = test.drop(drop_col_test, axis=1)

drop_col_train = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', '3SsnPorch']
train = train.drop(drop_col_train, axis=1)
""", language="python")

drop_col_test = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', '3SsnPorch']
test = test.drop(drop_col_test, axis=1)

drop_col_train = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', '3SsnPorch']
train = train.drop(drop_col_train, axis=1)

st.title('Target and feature')
# ## Target and feature

st.code("""
X = train.drop(['SalePrice'], axis=1)
y = np.log(train.SalePrice)
""", language="python")

X = train.drop(['SalePrice'], axis=1)
y = np.log(train.SalePrice)

st.title('Choosing numerical and categorical')
# ### Choosing numerical and categorical

st.code("""
numeric_features = make_column_selector(dtype_include=np.number)
categorical_features = make_column_selector(dtype_exclude=np.number)
""", language="python")

numeric_features = make_column_selector(dtype_include=np.number)
categorical_features = make_column_selector(dtype_exclude=np.number)

st.title('Pipeline imputer - scaler - encoder')

st.code("""
numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler())

categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                    OneHotEncoder(sparse=False, handle_unknown='ignore'))
""", language="python")

numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler())

categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                    OneHotEncoder(sparse=False, handle_unknown='ignore'))

st.title('Processing on categorical and numerical')


st.title('Pre - Processing on categorical and numerical')

st.code("""
transformer = make_column_transformer(
    (numerical_pipeline,numeric_features),
    (categorical_pipeline, categorical_features))
""", language="python")

transformer = make_column_transformer(
    (numerical_pipeline,numeric_features),
    (categorical_pipeline, categorical_features))


st.title('choosing model and apply pre processing')

st.code("""
model = make_pipeline(transformer, RandomForestRegressor())
""", language="python")

model = make_pipeline(transformer, Ridge())

st.write(model)


model.fit(X, y)
score = model.score(X, y)

st.write(score)


X_test = test.drop(columns = 'Id')
preds = model.predict(X_test)
preds = np.exp(preds)





final_df = pd.DataFrame(test.Id)
#final_df




final_df['Saleprice'] = pd.DataFrame(preds)




st.write(final_df)
#final_df











