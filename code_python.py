import pandas as pd
import re
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

#  load the dataset
id = "1t_bpJX7PC0dqZ5pEmaxM8wmDJteYmOXB" # google file ID
df = pd.read_csv("https://docs.google.com/uc?id=" + id, sep = ';')
df.head()

# do the cleaning
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

def clean_text(text):
    if isinstance(text, str):
        # delete websites
        text_noweb = re.sub('http\\S+', '', text)
        # delete numbers
        text_nonum = re.sub(r'\d+', '', text_noweb)
        # make a list of tokens
        try:
            text_token = word_tokenize(text_nonum, language='english')
        except IndexError:
            try:
                text_token = word_tokenize(text_nonum[1:], language='english')
            except:
                text_token = []

        # add words that don't contain special symbols
        clean_words = []
        for word in text_token:
            if word.isalnum():
                clean_words.append(word)

        # delete stopwords and make lowercase
        words_nostop = [word.lower() for word in clean_words if word.lower() not in stop_words]

        # get the stem of the word
        words_stem = [stemmer.stem(word) for word in words_nostop]
        
        return ' '.join(words_stem)
    else:
        return ''

df['title_clean'] = df.title.apply(lambda x: clean_text(x))
df.head()

# calculate TF-IDF
vectorizer = TfidfVectorizer(min_df=8)

X = df.title_clean
y, uniques = pd.factorize(df.topic)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.01, random_state = 0)

X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)
X_test = vectorizer.transform(X_test)

print(X_train.shape)

########################################################################
## Logistic Regression
########################################################################

## Multinomial classification
# do cross validation over the defined values of C
# alpha = 1
Cs = [100.0, 50.0, 10.0, 1.0, 0.1, 0.01, 0.001]
log1 = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=10000, multi_class='multinomial', solver='saga', n_jobs=-1, penalty='l1')
log1.fit(X_train, y_train)

print(log1.score(X_train, y_train))
print(log1.score(X_val, y_val))
print(log1.scores_)
print(log1.C_)

# alpha = 0.5
log05 = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=10000, multi_class='multinomial', solver='saga', n_jobs=-1, penalty='elasticnet')
log05.fit(X_train, y_train)

print(log05.score(X_train, y_train))
print(log05.score(X_val, y_val))
print(log05.scores_)
print(log05.C_)

# alpha = 0
log0 = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=10000, multi_class='multinomial', solver='saga', n_jobs=-1, penalty='l2')
log0.fit(X_train, y_train)

print(log0.score(X_train, y_train))
print(log0.score(X_val, y_val))
print(log0.scores_)
print(log0.C_)

## ---------------------------------------------------------------------------------------------
## One Vs Rest approach
# alpha = 1
Cs = (100, 50, 10, 1, 0.1, 0.01, 0.001) 
log1 = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=10000, multi_class='ovr', solver='saga', n_jobs=-1, penalty='l1')
log1.fit(X_train, y_train)

print(log1.score(X_train, y_train))
print(log1.score(X_val, y_val))
print(log1.scores_)
print(log1.C_)

# alpha = 0.5
log05 = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=10000, multi_class='ovr', solver='saga', n_jobs=-1, penalty='elasticnet')
log05.fit(X_train, y_train)

print(log05.score(X_train, y_train))
print(log05.score(X_val, y_val))
print(log05.scores_)
print(log05.C_)

# alpha = 0
log0 = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=10000, multi_class='ovr', solver='saga', n_jobs=-1, penalty='l2')
log0.fit(X_train, y_train)

print(log0.score(X_train, y_train))
print(log0.score(X_val, y_val))
print(log0.scores_)
print(log0.C_)

########################################################################
## Sparse SVM
########################################################################

## define grid params for grid search cross validation
grid_params = {
    'C': [1000, 100, 10, 1, 0.1, 0.01, 0.001],
}

# Lasso penalty
svm1 = LinearSVC(random_state=42, penalty='l1', dual=False, max_iter=10000)
svm_cv1 = GridSearchCV(svm1, param_grid=grid_params, n_jobs = -1, cv=4)
svm_cv1.fit(X_train, y_train)

print(svm_cv1.score(X_train, y_train))
print(svm_cv1.score(X_val, y_val))
print(svm_cv1.best_params_)
print(svm_cv1.best_score_)

# Ridge penalty
svm0 = LinearSVC(random_state=42, penalty='l2', dual=False, max_iter=10000)
svm_cv0= GridSearchCV(svm0, param_grid=grid_params, n_jobs = -1, cv=4)
svm_cv0.fit(X_train, y_train)

print(svm_cv0.score(X_train, y_train))
print(svm_cv0.score(X_val, y_val))
print(svm_cv0.best_params_)
print(svm_cv0.best_score_)