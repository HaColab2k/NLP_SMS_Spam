# TEXT PREPROCESSING
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
Performs text preprocessing on a given string of text. The purpose of this function is to clean and prepare the text data for further analysis or natural language processing tasks.
```Python
nopunc = [char for char in mess if char not in string.punctuation]
```
Creates a list nopunc containing characters from the input mess that are not in the list of punctuation characters from the string library.
```Python
nopunc = ''.join(nopunc)
```
Joins the characters in the nopunc list to form a single string without punctuation.
```Python
STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
```
Defines a custom list of stopwords to supplement the NLTK stopwords. It includes additional words like chat slang or internet abbreviations that you want to consider as stopwords.