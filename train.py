import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('water_potability.csv')
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'water_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
print("Done! Model save ho gaya")