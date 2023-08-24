#!/usr/bin/env python
# coding: utf-8

# # 1. Data Preprocessing

# In[1]:


import pandas as pd


# ### 1.1 Loading Data & initial data exploration

# In[2]:


data = pd.read_excel('C:\\Users\\hp\\Downloads\\customer_churn_large_dataset.xlsx')
data


# In[3]:


data.head(10)


# In[4]:


data.describe()


# In[5]:


data.dtypes


# In[6]:


data['Churn'].value_counts()


# In[7]:


data.nunique()


# ### 1.2 Handling missing data 

# In[8]:


data.isna().sum()


# ### 1.3 Encoding Categorical variable

# In[9]:


data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
data


# In[10]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['LocationNew'] = label_encoder.fit_transform(data['Location'])
data


# In[11]:


city_namesE = data['LocationNew'].apply(lambda x: str(x).split(',')[0].strip()).unique()
city_namesE


# In[12]:


city_names = data['Location'].apply(lambda x: str(x).split(',')[0].strip()).unique()
city_names


# # 2. Feature Engineering

# In[13]:


data.drop(['Name', 'Location'], axis=1, inplace=True)
data


# In[34]:


data['bill_X_GB']= data['Monthly_Bill']*data['Total_Usage_GB']
data['Bill_/_subLen']= data['Monthly_Bill']/data['Subscription_Length_Months']
data['subs_/_bill']= data['Subscription_Length_Months']/data['Monthly_Bill']
data['GB_/_bill']= data['Total_Usage_GB']/data['Monthly_Bill']
data['GB_X_subLen']= data['Total_Usage_GB']*data['Monthly_Bill']
data


# ### 2.1 Scaling & Normalization

# In[35]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
normalized_df


# ### 2.2 Finding Correlation

# In[36]:


corr_matrix = normalized_df.corr()
corr_matrix['Churn'].sort_values(ascending=False)


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


from pandas.plotting import scatter_matrix
attributes = ['Subscription_Length_Months','Monthly_Bill','Total_Usage_GB','Churn', 'LocationNew']
scatter_matrix(data[attributes], figsize = (10,10), alpha= 0.3)


# In[39]:


data.plot(kind='scatter',x='LocationNew',y='Churn', alpha=.9)


# In[40]:


city = data.loc[data['Churn']==1, 'LocationNew']
city.groupby(city).size()

# [     2            4          3         0          1]
# ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston']


# In[41]:


import numpy as np
print("Has infinite values:", np.any(np.isinf(normalized_df[['bill_X_GB','Bill_/_subLen','subs_/_bill']])))


# In[42]:


normalized_df.isna().sum()


# In[43]:


normalized_df.plot(kind='scatter',x='subs_/_bill',y='bill_X_GB', alpha=.9)


# In[44]:


corr_matrix_2 = normalized_df.corr()
corr_matrix_2['Churn'].sort_values(ascending=False) 


# ### 2.3 Splitting Train & Test data

# In[45]:


from sklearn.model_selection import train_test_split

X = normalized_df[['LocationNew','Monthly_Bill', 'LocationNew', 'Total_Usage_GB', 'bill_X_GB']]
y = normalized_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(len(X_train))
print(len(y_test))


# In[46]:


import numpy as np
print("Has infinite values:", np.any(np.isinf(X)))

max_value = np.max(X)
print("Maximum value in X:", max_value)


# # 3 Model Building

# ### 3.1 Logistic Regresion

# In[47]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ### 3.2 Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators= 100, random_state = 42)
model2.fit(X_train, y_train)
model2.score(X_test, y_test)


# ### 3.3 Neural Network

# In[49]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=X_scaled.shape[1]))

model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)


accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)


# ### 3.4 Model Performance

# In[50]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
confusion = confusion_matrix(y_test, y_pred_classes)
classification_rep = classification_report(y_test, y_pred_classes)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)


# # 4. Model Optimization

# ### 4.1 Fine Tuning

# In[51]:


from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score



model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=X_scaled.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

best_model = load_model('best_model.h5')

y_pred = best_model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)


# ### 4.3 Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier


# Function to create the ANN model
def create_ann(learning_rate=0.001):
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_dim=X_scaled.shape[1]))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier with adjustable parameters
keras_classifier = KerasClassifier(build_fn=create_ann, verbose=0)

# Perform stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = cross_val_score(keras_classifier, X_scaled, y, cv=cv, scoring='accuracy')

print("Cross-validation results:", cross_val_results)
print("Mean Accuracy:", cross_val_results.mean())
print("Standard Deviation:", cross_val_results.std())

# Define hyperparameters and their possible values
param_grid = {
    'epochs': [10, 13],
    'batch_size': [32, 64],
    'learning_rate': [0.001, 0.01]
}

# Perform grid search
grid = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, scoring='accuracy', cv=cv)
grid_result = grid.fit(X_scaled, y)

print("Best Score:", grid_result.best_score_)
print("Best Params:", grid_result.best_params_)



# In[ ]:




