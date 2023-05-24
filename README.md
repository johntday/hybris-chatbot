# Hybris Chatbot

## Purpose
An experimental AI chatbot featuring conversational memory. Designed to enable users to ask questions about 
"SAP Commerce Cloud (Hybris)" in natural language.

## Demo
Use the following link to try it out :
TBD

## How to Run Locally
Follow these steps to set up and run the python app locally :

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
Clone the repository :

```bash
git clone https://github.com/johntday/hybris-chatbot.git
```

Navigate to the project directory :

```bash
cd hybris-chatbot
```

Create a virtual environment :
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install the required dependencies in the virtual environment :

```bash
pip install -r requirements.txt
```

Launch the chat service locally :

```bash
streamlit run hybris-chatbot.py
```

Switch to use the local Faiss vector database :

```bash
TBD
```
