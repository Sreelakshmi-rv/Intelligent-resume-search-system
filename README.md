# Intelligent Resume Search System (NLP Class Project - Prototype)

## Overview
This repository contains a prototype of an Intelligent Resume Search System, developed as a class project for Natural Language Processing (NLP). The aim of this system is to provide an efficient way to search and filter resumes based on skills, keywords, or other criteria, leveraging NLP techniques.

This project demonstrates foundational concepts in building a text-based search application and serves as a proof-of-concept for more advanced systems.

## Features (Prototype)
* **Resume Processing:** Likely processes uploaded resumes (e.g., PDF, TXT) to extract relevant text.
* **Keyword Matching:** Allows users to input keywords to find matching resumes.
* **Basic Search Functionality:** Provides a mechanism to list or rank resumes based on search queries.
* **User Interface:** Built with Streamlit for an interactive web application experience.

## Technologies Used
* **Python:** The core programming language for the application logic and NLP processing.
* **Streamlit:** For rapidly building and deploying the interactive web user interface.
* **NLTK / SpaCy / TextBlob (Likely):** For NLP tasks such as tokenization, stemming/lemmatization, stop-word removal, or keyword extraction.
* **Pandas / NumPy:** For data handling and manipulation.
* **Other Libraries:** (Add any other specific libraries you used, e.g., for document parsing like `PyPDF2` or `python-docx`)

## How to Run Locally
To set up and run this prototype on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sreelakshmi-rv/Intelligent-resume-search-system.git](https://github.com/Sreelakshmi-rv/Intelligent-resume-search-system.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Intelligent-resume-search-system
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
4.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
5.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser, typically at `http://localhost:8501`.

## Live Prototype
You can access the live (prototype) application here:
[https://intelligent-resume-search-system-aghuusgwdudnks84tc5yfb.streamlit.app/](https://intelligent-resume-search-system-aghuusgwdudnks84tc5yfb.streamlit.app/)

## Future Improvements
As a prototype, there are several areas for potential enhancement:
* **Advanced NLP Models:** Implement more sophisticated models (e.g., word embeddings, semantic search, deep learning models) for better understanding of resume content and job descriptions.
* **Robust Document Parsing:** Improve handling of various resume formats (PDF, DOCX) and complex layouts.
* **Enhanced UI/UX:** Improve the user interface for better usability and visual appeal.
* **Scalability:** Optimize for handling a larger volume of resumes efficiently.
* **Error Handling & Edge Cases:** Add more robust error handling for unexpected inputs or data formats.
* **Database Integration:** Implement a database for persistent storage and efficient retrieval of resume data.

## Contact
* **GitHub:** [Sreelakshmi-rv](https://github.com/Sreelakshmi-rv)
