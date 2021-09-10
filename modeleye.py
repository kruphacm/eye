import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/Data%20of%20eyes.csv")
df['Symptom 1'] = df['Symptom 1'].map({'Cloudy vision/Blurred Vision':0.0,'Sensitivity to light':1.0,'Poor vision at night':2.0,'Double vision':3.0,'Eye pain':4.0,'Red-eye':5.0,'Vision Loss':6.0,'Dry or watery eyes':7.0,'burning or itching':8.0,'Tired':9.0,'Blurred vision for distant objects, near objects, or both':10.0,'Headache':11.0,'Irritation, itching':12.0,"Foreign body sensation eye discomfort":13.0,'Burning':14.0,"Itching":15.0,"Loss of vision":16.0,'Blurred vision':17.0,'Discharge or stickiness':18.0})
df['Symptom 2'] = df['Symptom 2'].map({'Cloudy vision/Blurred Vision':0.0,'Sensitivity to light':1.0,'Poor vision at night':2.0,'Double vision':3.0,'Eye pain':4.0,'Red-eye':5.0,'Vision Loss':6.0,'Dry or watery eyes':7.0,'burning or itching':8.0,'Tired':9.0,'Blurred vision for distant objects, near objects, or both':10.0,'Headache':11.0,'Irritation, itching':12.0,"Foreign body sensation eye discomfort":13.0,'Burning':14.0,"Itching":15.0,"Loss of vision":16.0,'Blurred vision':17.0,'Discharge or stickiness':18.0})
df['Symptom 3'] = df['Symptom 3'].map({'Cloudy vision/Blurred Vision':0.0,'Sensitivity to light':1.0,'Poor vision at night':2.0,'Double vision':3.0,'Eye pain':4.0,'Red-eye':5.0,'Vision Loss':6.0,'Dry or watery eyes':7.0,'burning or itching':8.0,'Tired':9.0,'Blurred vision for distant objects, near objects, or both':10.0,'Headache':11.0,'Irritation, itching':12.0,'Foreign body sensation eye discomfort':13.0,'Burning':14.0,"Itching":15.0,"Loss of vision":16.0,'Blurred vision':17.0,'Discharge or stickiness':18.0})
X = df.drop(['Disease','Treatment'], axis=1)
df['Disease'] = df['Disease'].map({'Cataract':0.0,'Glaucoma':1.0,'Eye Strain':2.0,'Refractive Errors':3.0,'Dry eye Syndrome':4.0,'Diabetic Retinopathy':5.0,'Conjunctivitis':6.0})
Y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

model1=LogisticRegression()
model1.fit(X,Y)
pickle.dump(model1, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))