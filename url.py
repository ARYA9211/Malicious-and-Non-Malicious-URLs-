import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset to inspect its structure and understand what kind of data it contains
file_path = 'url_dataset.csv'
url_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(url_data.head())



# Preprocessing: Convert labels to binary (1 for malicious, 0 for non-malicious)
label_encoder = LabelEncoder()
url_data['label'] = label_encoder.fit_transform(url_data['type'])

# Splitting data into features (urls) and labels (binary malicious/non-malicious)
X = url_data['url']
y = url_data['label']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Vectorizing the URLs using TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Check the shape of the TF-IDF vectors
print(X_train_tfidf.shape, X_test_tfidf.shape)




# Training Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Training Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Making predictions on the test set
y_pred_nb = nb_model.predict(X_test_tfidf)
y_pred_svm = svm_model.predict(X_test_tfidf)

# Generating classification reports for both models
nb_report = classification_report(y_test, y_pred_nb, target_names=label_encoder.classes_)
svm_report = classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_)

print("Naive Bayes Classification Report\n",nb_report)
print("SVM Classification Report\n",svm_report)
