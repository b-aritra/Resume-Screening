# Install first (already installed if you followed):
# pip install streamlit scikit-learn python-docx PyPDF2

import streamlit as st
import pickle
import docx  # For .docx files
import PyPDF2  # For PDF files
import re

# Load pre-trained model and vectorizers
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Classifier
tfidf = pickle.load(open('tfidf.pkl', 'rb'))    # TF-IDF Vectorizer
le = pickle.load(open('encoder.pkl', 'rb'))      # Label Encoder


# ----------- Text Cleaning Function -----------
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()


# ----------- Text Extraction Functions -----------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# ----------- File Upload Handling -----------
def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


# ----------- Prediction Function -----------
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    predicted_category = svc_model.predict(vectorized_text.toarray())  # Fix: Dense input!
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# ----------- Streamlit App -----------
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("📄 Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the **predicted job category**!")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted text from resume.")

            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Prediction
            st.subheader("🔮 Predicted Job Category")
            category = pred(resume_text)
            st.write(f"**The predicted category is:** 🎯 {category}")

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
