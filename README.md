#﻿Natural-Language-Processing (NLP)|SMS Spam

-This repository contains scripts for building an SMS spam detection system using Natural Language Processing (NLP) techniques.

## Install the required libraries:
```Python
pip install pandas numpy matplotlib seaborn nltk
```
## Importing Libraries:
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
```
This imports necessary libraries, such as pandas for data manipulation, numpy for numerical computations, matplotlib and seaborn for data visualization, and nltk for natural language processing.
## Setting Visualization Styles:
```Python
%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("dark_background")
```
These lines configure the visualization styles for your plots. %matplotlib inline ensures that plots are displayed inline in the Jupyter Notebook environment. The sns.set_style and plt.style.use lines set the background style of plots.
## Defining Sample Text Data:
```Python
simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']
```
This creates a list simple_train containing sample text data for training.

## Importing CountVectorizer and Fitting:
```Python
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(simple_train)
```
This imports the CountVectorizer class from scikit-learn and initializes an instance of it called vect. It then fits the vectorizer on the simple_train data. The vectorizer learns the vocabulary from this training data.
## Inspecting the Vocabulary:
```Python
vect.get_feature_names_out()
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/f49df1f9-6133-408c-ac46-3302d2916fef)




##Usage
-Prepare your SMS dataset in a CSV format.
-Open the Jupyter notebook sms_spam_detection.ipynb to run the analysis and train the classification models.
-Follow the code and comments in the notebook to understand the process of pre-processing, data visualization, and model training.
