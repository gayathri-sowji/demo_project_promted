# TARGET COLUMN: exam_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("students.csv")

# Drop irrelevant
df = df.drop(columns=["student_id"])

# Basic cleaning
for c in ['gender','course','internet_access',
          'sleep_quality','study_method',
          'facility_rating','exam_difficulty']:
    df[c] = df[c].str.lower().str.strip()

# Outlier clipping using IQR
def clip_iqr(col):
    q1, q3 = df[col].quantile([0.25,0.75])
    iqr = q3-q1
    df[col] = np.clip(df[col], q1-1.5*iqr, q3+1.5*iqr)

for col in ['study_hours','class_attendance','sleep_hours']:
    clip_iqr(col)

# Ordinal mapping
ordinal_maps = {
 'sleep_quality': {'poor':1,'average':2,'good':3},
 'facility_rating': {'low':1,'medium':2,'high':3},
 'exam_difficulty': {'easy':1,'moderate':2,'hard':3}
}

for k,v in ordinal_maps.items():
    df[k+"_enc"] = df[k].map(v)

# Scaling numerical
scaler = StandardScaler()
df[['age','study_hours','class_attendance','sleep_hours']] = \
 scaler.fit_transform(df[['age','study_hours','class_attendance','sleep_hours']])

# Export
df.to_csv("cleaned_students.csv", index=False)
print("Exported cleaned_students.csv")
