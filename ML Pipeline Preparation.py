#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[32]:


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import multioutput
import pickle
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD


# In[15]:


# load data from database
engine = create_engine('sqlite:///DisasterResponseCat.db')
df = pd.read_sql_table('df',engine)
X = df['message']
Y = df.iloc[:,4:]
df.head()
category_names = Y.columns


# ### 2. Write a tokenization function to process your text data

# In[16]:


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    words = word_tokenize(text)
    stemmed = [PorterStemmer().stem(w) for w in words]
    words_lemmatized = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
    return words_lemmatized
#     pass


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[17]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
pipeline.fit(X_train, Y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[19]:


Y_pred = pipeline.predict(X_test)
for i in range(36):
    print(Y_test.columns[i], ':')
    print(classification_report(Y_test.iloc[:,i], Y_pred[:,i], target_names=category_names), '...................................................')


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[20]:


# def average_f1(Y_real, Y_predictions, avg = 'macro'):
#     '''
#     Computes average f1-score across all categories:
    
#     inputs:
#     Y_real: correct classifications
#     Y_predictions: predicted classifications
#     avg: Default == 'macro'. See f1_score doc for other options
    
#     Return:
#     accuracy
#     '''
    
#     f1 = []
    
#     # iterate
#     for i, category in enumerate(Y_real):
#         f1.append(f1_score(Y_real[category], Y_predictions[:, i], average = avg))
        
    
#     return np.mean(f1)


# In[21]:


pipeline.get_params()


# In[22]:


# parameters = { 'vect__max_df': (0.75, 1.0),
#                 'clf__estimator__n_estimators': [10, 20],
#                 'clf__estimator__min_samples_split': [2, 5]
#               }

parameters = {
    'tfidf__use_idf':[True, False],
    'clf__estimator__n_estimators':[10],
#     'clf__estimator__learning_rate':[1.0,1.5,2.0]
}
# scorer = make_scorer(fbeta_score, beta=2)
# scorer = make_scorer (f1_scorer_eval)
# scorer = make_scorer(average_f1)
# cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer,verbose=7)
cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, verbose=0)
# print(Y_train)


# In[23]:


cv.fit(X_train,Y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[24]:


Y_pred_cv = cv.predict(X_test)


# In[27]:


for i in range(36):
    print(Y_test.columns[i], ':')
    print(classification_report(Y_test.iloc[:,i], Y_pred_cv[:,i], target_names=category_names), '...................................................')


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[29]:


vect_check = CountVectorizer(tokenizer=tokenize).fit_transform(X_test)
vect_check.shape


# In[30]:


tfidf_check = TfidfTransformer().fit_transform(vect_check)
tfidf_check.shape


# In[33]:


svd_check = TruncatedSVD(n_components=100, random_state=42).fit_transform(tfidf_check)
svd_check.shape


# In[35]:


pipeline_alt = Pipeline([
        ('vect' , CountVectorizer(tokenizer=tokenize)),
        ('tfidf' , TfidfTransformer()),
        ('svd',TruncatedSVD(n_components=100, random_state=42)),
        ('clf' , MultiOutputClassifier(RandomForestClassifier(random_state = 42)))
])


# In[ ]:


pipeline_alt.fit(X_train, Y_train)


# In[ ]:





# In[ ]:


Y_pred_alt = pipeline_alt.predict(X_test)
for i in range(36):
    print(Y_pred_alt.columns[i], ':')
    print(classification_report(Y_pred_alt.iloc[:,i], Y_pred[:,i], target_names=category_names), '...................................................')


# ### 9. Export your model as a pickle file

# In[20]:


filename = 'classifier.sav'
pickle.dump(pipeline, open(filename, 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




