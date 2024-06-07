import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Function to preprocess text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# Function to load model and make predictions
def load_model(data_file):
    vectorization = TfidfVectorizer()
    df = pd.read_csv(data_file)
    df["text"] = df["text"].apply(wordopt)
    x = df["text"]
    y = df["class"]
    xv = vectorization.fit_transform(x)
    
    LR = LogisticRegression()
    DT = DecisionTreeClassifier()
    GBC = GradientBoostingClassifier(random_state=0)
    RFC = RandomForestClassifier(random_state=0)
    
    models = {"Logistic Regression": LR, "Decision Tree": DT, "Gradient Boosting": GBC, "Random Forest": RFC}
    
    for model in models.values():
        model.fit(xv, y)
    
    return vectorization, models

# Function to predict using loaded model
def predict_news(news, vectorization, models):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x = new_def_test["text"]
    new_xv = vectorization.transform(new_x)
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(new_xv)[0]
    
    return predictions

# UI design
def main():
    st.title("Fake News Detection")
    st.write("Enter your news text below:")
    news = st.text_area("News Text", "")
    
    data_file = "manual_testing.csv"  # Change this to the desired dataset file name
    
    if st.button("Detect"):
        try:
            vectorization, models = load_model(data_file)
            predictions = predict_news(news, vectorization, models)
            
            st.subheader("Prediction Results:")
            for model, prediction in predictions.items():
                result = "Fake News" if prediction == 0 else "Not Fake News"
                st.write(f"{model}: {result}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
