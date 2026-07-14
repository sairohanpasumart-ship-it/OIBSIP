import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('Iris.csv')
le = LabelEncoder()
scaler = StandardScaler()
model = SVC()
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = le.fit_transform(df['Species'])
model.fit(scaler.fit_transform(X), y)
sample = scaler.transform([[5.9,3.0,5.1,1.8]])
print("Predicted:", le.inverse_transform(model.predict(sample))[0])
