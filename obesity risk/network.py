import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Optional: Calculate BMI and add it as a new feature if not already included
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
test_df['BMI'] = test_df['Weight'] / (test_df['Height'] ** 2)

label_encoder = LabelEncoder()
df['NObeyesdad_encoded'] = label_encoder.fit_transform(df['NObeyesdad'])
y = df['NObeyesdad_encoded']

# Define features and target
X = df.drop(['NObeyesdad','NObeyesdad_encoded', 'id'], axis=1)  # assuming 'id' column is present and should be excluded
id = test_df['id']
X_actual_test = test_df.drop(['id'], axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, X.select_dtypes(include=['object', 'bool']).columns)
    ])

# Define the model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Training the model
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)


# Making the Confusion Matrix
print(classification_report(y_test, y_pred))

# Predicting the Test set results
y_actual_pred = clf.predict(X_actual_test)

# Convert the numerical predictions back to original categorical labels
y_test_pred_labels = label_encoder.inverse_transform(y_actual_pred)

# Combine these labels with the 'id' column from the test dataset
test_predictions = pd.DataFrame({
    'id': test_df['id'],
    'NObeyesdad': y_test_pred_labels
})

# Save the combined data to a new CSV file
test_predictions.to_csv('test_predictions.csv', index=False)



