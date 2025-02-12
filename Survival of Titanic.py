import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_excel(r'titanic.xlsx')
print(df.head())

# Separate independent and dependent features
inputs = df.drop('Survived', axis="columns")
output = df['Survived']


# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(inputs, output)

scr = model.score(inputs,output)
print(scr)

#Predict
op = model.predict([[3,35]])
print("Survived?" , op)

