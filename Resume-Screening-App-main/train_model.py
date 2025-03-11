import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Sample resume dataset
resumes = [
    "Software engineer with experience in Python and ML.",
    "Marketing expert skilled in SEO and social media.",
    "Data analyst proficient in SQL and Tableau.",
    "Web developer with expertise in React and Node.js."
]
categories = ["Software Engineer", "Marketing", "Data Analyst", "Web Developer","Data Analyst"]

# Convert text data to numerical features using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(resumes)

# Train an SVM classifier
svc_model = SVC(probability=True)
svc_model.fit(X, categories)

# Save the model and vectorizer
with open("clf.pkl", "wb") as f:
    pickle.dump(svc_model, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model training complete! Files saved as `clf.pkl` and `tfidf.pkl`.")
