from pycaret.classification import *
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('transaction_dataset.csv', index_col=0)
df.sample(3)

df.shape

# drop first two columns (Index, Adress)
df = df.iloc[:, 2:]


setup(df, target="FLAG", session_id=85)

compare_models()

df.info()

df.select_dtypes(include=['float', 'int']).describe()

df['FLAG'].value_counts()

# fig = px.pie(df, values=df['FLAG'].value_counts().values, names=df['FLAG'].value_counts() ,
#              title='Target distribution of being Fraud or not', color_discrete_sequence=px.colors.sequential.RdBu)
# fig.show()

print(f'Percentage of non-fraudulent instances : {len(df.loc[df["FLAG"]==0])/len(df["FLAG"])*100}')
print(f'Percentage of fraudulent instances : {len(df.loc[df["FLAG"]==1])/len(df["FLAG"])*100}')
