import streamlit as st
import pickle
import re
import nltk

# nltk.download('punkt')
# nltk.download('stopswords')

# loading models
clf = pickle.load(open('knn_classifier.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove hashtags and mentions
    text = re.sub(r'#\w+|\@\w+', '', text)
    
    # Remove special characters and punctuations (excluding spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    st.title("Resume Screening App")
    
    upload_file = st.file_uploader('Upload Resume', type = ['pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')

        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
            # print("Ho gya decode !!")

        cleaned_res = clean_text(resume_text)
        cleaned_res = tfidf.transform([cleaned_res])

        pred_id = clf.predict(cleaned_res)[0]

        st.write(pred_id)

        cate_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        cate_name = cate_mapping.get(pred_id, "Unknown")

        # print("Prediction Category :", cate_name + " /// "+" Prediction Id : ",  pred_id)
        st.write(cate_name)


if __name__ == '__main__':
    main()


# st.title("duihuddhi")