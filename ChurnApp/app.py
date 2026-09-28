#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import and load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# sklearn tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the dataset
dataset_path = (
    Path(__file__).resolve().parent.parent
    / "Assignment Problem Statement 4"
    / "Assignment PS4"
    / "customer_churn_dataset.csv"
)
df = pd.read_csv(dataset_path)

# Sanity Check
print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.info())


# In[2]:


# data visualization# 1. Countplot for Churn target
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribution of Customer Churn')
plt.show()

# 2. Heatmap for numeric feature correlation
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


# In[3]:


# data cleaning
# Drop unique identifier
df.drop('customerID', axis=1, inplace=True)

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn' in numeric_cols: numeric_cols.remove('Churn')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numeric missing values with 'mean'
num_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Impute categorical missing values with 'mode'
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("Data Cleaning Complete. Missing values:", df.isnull().sum().sum())


# In[4]:


# Encoding converting words into numbers
# Use pd.get_dummies to convert text to numbers
df_final = pd.get_dummies(df, drop_first=True)

print(f"Features after encoding: {df_final.shape[1]}")
df_final.head()


# In[5]:


# Split data into Features (X) and Target (y)
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']

# 80/20 Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


X_train_scaled


# In[7]:


X_test_scaled


# In[8]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[9]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(5)
knn.fit(X_train,y_train)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)


# In[12]:


# A dictionary to store our 4 required models for easy looping
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

# The Loop: Trains each model and prints its accuracy
for name, model in models.items():
    model.fit(X_train_scaled, y_train) # The actual "learning" process
    score = model.score(X_test_scaled, y_test) # Checking accuracy on test data
    print(f"{name} accuracy: {score:.4f}")


# In[13]:


from sklearn.model_selection import GridSearchCV

# 1. Logistic Regression Tuning (Tuning the 'C' parameter for regularization)
param_grid_lr = {'C': [0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5)
grid_lr.fit(X_train_scaled, y_train)

# 2. Decision Tree Tuning (Tuning the 'max_depth' to prevent overfitting)
param_grid_dt = {'max_depth': [3, 5, 10, None]}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_dt.fit(X_train_scaled, y_train)

# 3. KNN Tuning (Tuning the number of neighbors 'n_neighbors')
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train_scaled, y_train)

# 4. Random Forest Tuning (Tuning the number of trees 'n_estimators')
param_grid_rf = {'n_estimators': [50, 100, 200]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train)

print("Best Parameters Found:")
print(f"LR: {grid_lr.best_params_}")
print(f"DT: {grid_dt.best_params_}")
print(f"KNN: {grid_knn.best_params_}")
print(f"RF: {grid_rf.best_params_}")

# Update our models dictionary with the best-tuned versions
tuned_models = {
    "Logistic Regression": grid_lr.best_estimator_,
    "Decision Tree": grid_dt.best_estimator_,
    "K-Nearest Neighbors": grid_knn.best_estimator_,
    "Random Forest": grid_rf.best_estimator_
}


# In[14]:


# 1. Create a list to store final results
final_comparison = []

# 2. Loop through the Tuned Models
for name, model in tuned_models.items():
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    # Get probabilities for AUC-ROC
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Append results
    final_comparison.append([name, acc, prec, rec, f1, roc_auc])

# 3. Create Comparison DataFrame
eval_df = pd.DataFrame(final_comparison, 
                       columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

# 4. Display the table (sorted by F1-Score)
print("--- Final Evaluation Metrics (Tuned Models) ---")
print(eval_df.sort_values(by='F1-Score', ascending=False).to_string(index=False))

# 5. Plot the Comparison Chart
eval_df.set_index('Model').plot(kind='bar', figsize=(12, 7))
plt.title('Comparison of Tuned Model Performance')
plt.ylabel('Score (0.0 - 1.0)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[15]:


# 1. Pick a sample customer from the scaled test data
# We use .reshape(1, -1) because the model expects a 2D array (a table with 1 row)
new_customer_data = X_test_scaled[0].reshape(1, -1) 

# 2. Use our best model (e.g., Random Forest) to make the prediction
# You can change "Random Forest" to "Logistic Regression" to compare
prediction = tuned_models["Random Forest"].predict(new_customer_data)

# 3. Get the probability (Percentage chance of leaving)
probability = tuned_models["Random Forest"].predict_proba(new_customer_data)[0][1]

print(f"--- Prediction Results ---")
if prediction[0] == 1:
    print("Result: High Risk of Churn")
else:
    print("Result: Likely to Stay (Loyal)")

print(f"Churn Probability: {probability:.2%}")


# In[16]:


# 1. Create a "Template" for a new customer using the columns from our training data
# This creates a row of all zeros to start with
new_customer_raw = pd.DataFrame(0, index=[0], columns=X.columns)

# 2. Manually fill in the values for this specific customer
# (Fill only what you need; everything else remains 0)
new_customer_raw['tenure'] = 3             # 12 months with the company
new_customer_raw['SeniorCitizen'] = 0       # Not a senior
new_customer_raw['gender_Male'] = 1         # Is Male
new_customer_raw['Partner_Yes'] = 0        # Has a partner
new_customer_raw['InternetService_Fiber optic'] = 1 # USES Fiber Optic
new_customer_raw['PhoneService_Yes'] = 0    # Has phone service

# 3. Scale the input using the same scaler we used for training
# This converts the raw numbers into the "scaled" format the model understands
new_customer_scaled = scaler.transform(new_customer_raw)

# 4. Final Prediction using the best model
prediction = tuned_models["Random Forest"].predict(new_customer_scaled)
probability = tuned_models["Random Forest"].predict_proba(new_customer_scaled)[0][1]

print("--- Individual Prediction Result ---")
if prediction[0] == 1:
    print("Prediction: CUSTOMER WILL CHURN (Leave)")
else:
    print("Prediction: CUSTOMER WILL STAY (Loyal)")

print(f"Churn probability: {probability:.2%}")


# In[17]:


import joblib

# 1. Save the best model (Random Forest) 
# This 'freezes' the brain of your model into a file
joblib.dump(tuned_models["Random Forest"], 'churn_model.pkl')

# 2. Save the Scaler 
# This ensures new data is mathematically scaled exactly like your training data
joblib.dump(scaler, 'scaler.pkl')

# 3. Save the Features (Column Names)
# This prevents the "10 vs 14 features" error by remembering the correct order
joblib.dump(X.columns.tolist(), 'features.pkl')

print("Deployment files successfully exported!")


# In[21]:


from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load the pre-trained model and preprocessing tools
# Ensure these .pkl files are in your 'ChurnApp' folder
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # 2. Initialize a blank DataFrame with 0s for ALL features in your dataset
        # This ensures the model only 'listens' to the high-impact variables we set next.
        input_df = pd.DataFrame(0, index=[0], columns=features)

        # 3. Map the HIGH-IMPACT Prediction Drivers
        # A. Tenure: The #1 predictor (52% importance)
        if 'tenure' in features:
            input_df['tenure'] = float(data['tenure'])
        
        # B. SeniorCitizen: Key demographic driver (6.5% importance)
        if 'SeniorCitizen' in features:
            input_df['SeniorCitizen'] = int(data['SeniorCitizen'])

        # C. Family Status: Partner and Dependents (~4.3% importance each)
        if data.get('Partner') == 'Yes' and 'Partner_Yes' in features:
            input_df['Partner_Yes'] = 1
            
        if data.get('Dependents') == 'Yes' and 'Dependents_Yes' in features:
            input_df['Dependents_Yes'] = 1

        # D. Sticky Services: Online Security & Online Backup (~3.2% importance)
        if data.get('OnlineSecurity') == 'Yes' and 'OnlineSecurity_Yes' in features:
            input_df['OnlineSecurity_Yes'] = 1
        elif data.get('OnlineSecurity') == 'No internet service' and 'OnlineSecurity_No internet service' in features:
            input_df['OnlineSecurity_No internet service'] = 1

        if data.get('OnlineBackup') == 'Yes' and 'OnlineBackup_Yes' in features:
            input_df['OnlineBackup_Yes'] = 1
        elif data.get('OnlineBackup') == 'No internet service' and 'OnlineBackup_No internet service' in features:
            input_df['OnlineBackup_No internet service'] = 1

        # 4. Standardize and Calculate Probability
        scaled_input = scaler.transform(input_df)
        
        # Predict the discrete label (0 or 1)
        prediction = model.predict(scaled_input)[0]
        
        # Calculate the specific Churn Probability (class 1)
        # This aligns with the PS4 goal of identifying 'at-risk' customers
        probability_array = model.predict_proba(scaled_input)
        churn_prob = round(probability_array[0][1] * 100, 2)

        # Define status message
        status = "YES - High Churn Risk" if prediction == 1 else "NO - Low Churn Risk"

        return render_template('result.html', 
                               prediction=status, 
                               churn_prob=churn_prob)

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == '__main__':
    # Use port 8080 to avoid Windows 10/11 system port conflicts
    app.run(debug=True, port=8080)


# In[20]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Generate the Confusion Matrix for your best model (e.g., Random Forest)
y_pred = tuned_models["Random Forest"].predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

# 2. Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix: Prediction vs. Reality')
plt.show()

# 3. Print the detailed report (Precision, Recall, F1)
print("Detailed Performance Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




