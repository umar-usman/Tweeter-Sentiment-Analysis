import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#Reading the dataset
tweets = pd.read_csv("Tweets.csv")

#Removing the Irrelevant data
del tweets['airline_sentiment_confidence']
del tweets['negativereason_confidence']
del tweets['airline_sentiment_gold']
del tweets['name']
del tweets['negativereason']
del tweets['negativereason_gold']
del tweets['retweet_count']
del tweets['tweet_coord']
del tweets['tweet_created']
del tweets['tweet_location']
del tweets['user_timezone']
del tweets['tweet_id']
del tweets['airline']

#Rearranged the class column to the last
tweets = tweets[['text','airline_sentiment']]

#Data Exploration through Bar Graphs
Mood_count=tweets['airline_sentiment'].value_counts()
Index = [1,2,3]
plt.bar(Index,Mood_count)
plt.xticks(Index,['negative','neutral','positive'],rotation=45)
plt.ylabel('Count')
plt.xlabel('Sentiments')
plt.title('Sentiment Count')

#Converting categorical data into numerical
tweets.airline_sentiment.replace(['positive', 'negative','neutral'], [1, 0, 2], inplace=True)

#Tfidf Vectorized features
vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='word', max_df= 0.25, min_df= 12, stop_words= 'english')
vectorized_features = vectorizer.fit_transform(tweets['text'])

#Splitting the dataset into Training set and Testing set.
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(vectorized_features, tweets.airline_sentiment, train_size=0.7,test_size = 0.3, random_state = 10)

#Fitting the Random Forest Model and Predicting
clf1 = RandomForestClassifier(max_depth=2, random_state=0)
clf1.fit(X_train_count, y_train_count)
pred = clf1.predict(X_test_count)
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = clf1 , X = X_train_count, y = y_train_count, cv = 10)

#Fitting the AdaBoost Model and Predicting
clf2 = AdaBoostClassifier()
clf2.fit(X_train_count, y_train_count)
pred = clf2.predict(X_test_count)
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = clf2 , X = X_train_count, y = y_train_count, cv = 10)

#Fitting the Naive Bayes Model and predicting.
model_count_NB = BernoulliNB(alpha=0.05)
model_count_NB.fit(X_train_count, y_train_count)
predictions_count_NB = model_count_NB.predict(X_test_count)

#Fitting the SVC Model and predicting.
model_count_SVC = SVC(kernel = 'linear', random_state = 0)
model_count_SVC.fit(X_train_count, y_train_count)
predictions_count_SVM = model_count_SVC.predict(X_test_count)

# cross validation with kfold = 10
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = model_count_NB , X = X_train_count, y = y_train_count, cv = 10)

# cross validation with kfold = 10
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = model_count_SVC , X = X_train_count, y = y_train_count, cv = 10)

#Ensemble of SVM and NB
eclf = VotingClassifier(estimators=[('nb', model_count_NB), ('svc', model_count_SVC)],voting='hard')

#Ensembling predictions
eclf.fit(X_train_count, y_train_count)
predictions_count = eclf.predict(X_test_count)

# cross validation with kfold = 10
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = eclf, X = X_train_count, y = y_train_count, cv = 10)
print ('Ensemble Mean Accuracy',accuracies.max()*100)
