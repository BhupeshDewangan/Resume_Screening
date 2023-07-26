import streamlit as st
import pickle
import re
import nltk

import base64
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

# nltk.download('punkt')
# nltk.download('stopswords')

# loading models
clf = pickle.load(open('knn_classifier.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


PAGE_TITLE = "📄 Resume Analyser 💻 !!"

st.set_page_config(page_title=PAGE_TITLE)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_xnbikipz.json"
lottie_hello = load_lottieurl(lottie_url_hello)

st_lottie(lottie_hello, key="hello")

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

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def home():
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
        st.header(cate_name)

        if upload_file is not None:
            # with st.spinner('Uploading your Resume....'):
            #     time.sleep(4)
            save_image_path = './Uploaded_Resumes/' + upload_file.name
            with open(save_image_path, "wb") as f:
                f.write(upload_file.getbuffer())

                show_pdf(save_image_path)


def dashboard():

    st.header("Showing Dashboard")

    df = pd.read_csv('UpdatedResumeDataSet.csv')

    st.subheader("Countplot for Dataset")
    # Create the countplot
    plt.figure(figsize=(15, 10))
    fig, ax = plt.subplots()
    sns.countplot(x=df['Category'])
    plt.xticks(rotation = 90)

    # Display the countplot using Streamlit
    st.pyplot(fig)



    st.subheader("Piechart for Dataset")
    # PIE CHART ------------------
    labels = df['Category'].unique()
    counts = df['Category'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.

    # Display the pie chart using Streamlit
    st.pyplot(fig)
    

def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Go to:", ("Home", "Dashboard"))

    if selected_page == "Home":
        home()
    elif selected_page == "Dashboard":
        dashboard()
    

if __name__ == '__main__':
    main()


# st.title("duihuddhi")