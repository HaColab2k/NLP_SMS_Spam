# NATURAL-LANGUAGE-PROCESSING (NLP)|SMS Spam

This repository contains scripts for building an SMS spam detection system using Natural Language Processing (NLP) techniques.

## 1. Install the required libraries:
```Python
pip install pandas numpy matplotlib seaborn nltk 
pip install -U scikit-learn
```
## 2. Exploratory Data Analysis (EDA)
### 2.1. Importing Libraries:
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
```
This imports necessary libraries, such as pandas for data manipulation, numpy for numerical computations, matplotlib and seaborn for data visualization, and nltk for natural language processing.
### 2.2. Setting Visualization Styles:
```Python
%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("dark_background")
```
These lines configure the visualization styles for your plots. %matplotlib inline ensures that plots are displayed inline in the Jupyter Notebook environment. The sns.set_style and plt.style.use lines set the background style of plots.
### 2.3. Defining Sample Text Data:
```Python
simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']
```
This creates a list simple_train containing sample text data for training.

### 2.4. Importing CountVectorizer and Fitting:
```Python
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(simple_train)
```
This imports the CountVectorizer class from scikit-learn and initializes an instance of it called vect. It then fits the vectorizer on the simple_train data. The vectorizer learns the vocabulary from this training data.
### 2.5. Inspecting the Vocabulary:
```Python
vect.get_feature_names_out()
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/f49df1f9-6133-408c-ac46-3302d2916fef)

This line retrieves the list of feature names (words) learned by the vectorizer. The get_feature_names_out() function returns the unique words in the vocabulary.
### 2.6. Transforming Text Data to Document-Term Matrix (DTM):
```Python
simple_train_dtm = vect.transform(simple_train)
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/3558991b-a363-439d-9606-2db9549b14da)

This line transforms the simple_train text data into a Document-Term Matrix (DTM) using the learned vocabulary. The DTM represents the occurrence of words in each document.

### 2.7. Converting DTM to Array:
```Python
simple_train_dtm.toarray()
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/69119bff-a585-42aa-9b3b-b50f03c973e7)

This line converts the DTM to a dense array format, which shows the count of each word in each document. It's a numeric representation suitable for machine learning algorithms.
### 2.8. Creating a DataFrame from the Document-Term Matrix:
```Python
df = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names_out())
df
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/a113225d-5f00-46d0-9925-8470caa29592)

This code creates a DataFrame df from the dense document-term matrix, where each row corresponds to a document and each column corresponds to a word in the vocabulary.
### 2.9. Checking the Type of Document-Term Matrix:
```Python
print(type(simple_train_dtm))
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/393b0b5a-5675-4b60-bf30-a8e25f4534b3)
### 2.10. Inspecting the Sparse Matrix Contents:
```Python
print(simple_train_dtm)
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/f783d6b8-9aaf-4e7b-9dd3-d9dabc21958e)

### Transforming Testing Data and Creating a DataFrame:
```Python
simple_test = ["please don't call me"]
simple_test_dtm = vect.transform(simple_test)
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names_out())
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/9ab02628-2f4a-4eac-9a75-cd3d88bdaf74)
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/eb86b731-0b08-43b4-a3c7-35fb7c6d8eca)

This code transforms a new testing dataset, simple_test, into a document-term matrix using the existing vocabulary. It then creates a DataFrame from the dense matrix, which represents the transformed testing data.
### 2.11. Reading and Preprocessing Data from a CSV File:
```Python
sms = pd.read_csv("spam.csv", encoding='latin-1')
sms.dropna(how="any", inplace=True, axis=1)
sms.columns = ['label', 'message']
```
This code reads data from a CSV file named "spam.csv" and sets the column names to "label" and "message". It drops any columns with missing values.

![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/8eb210be-887f-4a22-beeb-ff43d6c0a274)

### 2.12. Descriptive Analysis and Visualization:
```Python
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
sms['message_len'] = sms.message.apply(len)
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/ccd0c658-ba9d-42e7-b172-f21a3aa9c66f)
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/44f94817-06bf-4a7a-beeb-bdb2020e5f0d)

```Python
plt.figure(figsize=(6, 4))
sms[sms.label=='ham'].message_len.plot(bins=25, kind='hist', color='blue', 
                                       label='Ham messages', alpha=0.7)
sms[sms.label=='spam'].message_len.plot(kind='hist', color='red', 
                                       label='Spam messages', alpha=1)
plt.legend()
plt.xlabel("Message Length")
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/58d745bd-8473-4b2b-b63c-73a968e5042b)

This code adds new columns to the DataFrame: "label_num" (numerical label) and "message_len" (message length). It then creates a histogram to visualize the distribution of message lengths for both ham and spam messages.

### 2.13. Descriptive Statistics and Individual Message Inspection:
```Python
sms[sms.label=='ham'].describe()
sms[sms.label=='spam'].describe()
sms[sms.message_len == 910].message.iloc[0]
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/3de188e3-ff75-44fa-b3ca-53ac98bad0c4)

This code calculates descriptive statistics for message lengths separately for ham and spam messages. The last line retrieves the content of a specific message with a length of 910 characters.
## 3. Text Preprocessing
```Python
import string
from nltk.corpus import stopwords

def text_process(mess):
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
```
```Python
text_process(mess)
``` 
performs text preprocessing on a given string of text. The purpose of this function is to clean and prepare the text data for further analysis or natural language processing tasks.
```Python
nopunc = [char for char in mess if char not in string.punctuation]
```
creates a list nopunc containing characters from the input mess that are not in the list of punctuation characters from the string library.
```Python
nopunc = ''.join(nopunc)
```
joins the characters in the nopunc list to form a single string without punctuation.
```Python
STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
```
defines a custom list of stopwords to supplement the NLTK stopwords. It includes additional words like chat slang or internet abbreviations that you want to consider as stopwords.
## Usage
-Prepare your SMS dataset in a CSV format.
-Open the Jupyter notebook sms_spam_detection.ipynb to run the analysis and train the classification models.
-Follow the code and comments in the notebook to understand the process of pre-processing, data visualization, and model training.
