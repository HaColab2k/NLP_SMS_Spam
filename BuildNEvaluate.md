# BUILDING AND EVALUATING A MODEL 
## Training Multinomial Naive Bayes Classifier
```Python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
%time nb.fit(X_train_dtm, y_train)
```
Fit the classifier on the training data X_train_dtm (presumably the Document-Term Matrix) and the corresponding labels y_train. The %time command is used to measure the execution time of the fitting process.
## Making Predictions and Calculating Accuracy and Confusion Matrix:
```Python
y_pred_class = nb.predict(X_test_dtm)
print("=======Accuracy Score===========")
print(metrics.accuracy_score(y_test, y_pred_class))
print("=======Confusion Matrix===========")
metrics.confusion_matrix(y_test, y_pred_class)
```
![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/984b818c-a9ff-406d-bb57-2239ef51793a)

Making predictions using the trained Naive Bayes classifier on the test data X_test_dtm. The accuracy of the predictions is calculated using metrics.accuracy_score() and printed. The confusion matrix is calculated using metrics.confusion_matrix() to show the true positive, true negative, false positive, and false negative predictions.
# Comparing Predictions to Actual Data:
X_test[y_pred_class > y_test]
X_test[y_pred_class < y_test]
X_test[4949]

![image](https://github.com/HaColab2k/NLP_SMS_Spam/assets/127838132/f069f7f4-87d8-401b-a86e-bf51176af522)

## Calculating ROC AUC Score:
```Python
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
metrics.roc_auc_score(y_test, y_pred_prob)
```
The predicted probabilities of the positive class are calculated using predict_proba(), and the ROC AUC score is computed using metrics.roc_auc_score()

## Using a Pipeline with Text Preprocessing:
```Python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()), 
                 ('tfid', TfidfTransformer()),  
                 ('model', MultinomialNB())])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```
Sets up a pipeline that includes a CountVectorizer, a TfidfTransformer, and a MultinomialNB classifier. The pipeline applies text preprocessing steps (like counting word occurrences and applying TF-IDF transformation) along with the classifier. The pipeline is then fitted on the training data and used to make predictions on the test data. Accuracy and confusion matrix are printed similarly to the previous case.
