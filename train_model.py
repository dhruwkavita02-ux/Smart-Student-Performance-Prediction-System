import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Random dataset
np.random.seed(42)

rows = 500

data = pd.DataFrame({

    "attendance": np.random.randint(50,101,rows),
    "internal_marks": np.random.randint(30,100,rows),
    "assignment_marks": np.random.randint(30,100,rows),
    "study_hours": np.random.randint(1,8,rows),
    "previous_percentage": np.random.randint(40,100,rows)

})

# Target

performance=[]

for i in range(rows):

    score=(

        data.loc[i,"attendance"]*0.20+
        data.loc[i,"internal_marks"]*0.30+
        data.loc[i,"assignment_marks"]*0.20+
        data.loc[i,"study_hours"]*5+
        data.loc[i,"previous_percentage"]*0.30

    )

    if score>=85:
        performance.append("Excellent")

    elif score>=70:
        performance.append("Good")

    elif score>=55:
        performance.append("Average")

    else:
        performance.append("Poor")


data["performance"]=performance

X=data.drop("performance",axis=1)
y=data["performance"]

encoder=LabelEncoder()
y=encoder.fit_transform(y)

scaler=StandardScaler()
X=scaler.fit_transform(X)

model=RandomForestClassifier()
model.fit(X,y)

os.makedirs("model",exist_ok=True)

joblib.dump(model,"model/model.pkl")
joblib.dump(scaler,"model/scaler.pkl")
joblib.dump(encoder,"model/encoder.pkl")

print("✅ Model Trained Successfully!")