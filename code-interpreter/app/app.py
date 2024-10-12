import streamlit as st
import requests
import json

def run_query(input: str, task: str = 'instruct', target_lang: str = None):
    try:
        payload = {"query": input}
        if task == 'translate':
            payload = {"query": input, "target_lang": target_lang}
        response = requests.post(f"http://127.0.0.1:8080/domino/copilot/models/codey:{task}", data=json.dumps(payload)).json()
        answer = response['answer']
        code_exec = response['sandbox-output']
    except Exception as e:
        raise Exception(str(e))
    return answer, code_exec


answer, code_exec = '', ''
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 950px; /* Adjust the max-width value to your preference */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Code Interpreter")

# Code input
code_input = st.text_area("Enter your query here:", height=200)
col1, col2, col3, col4, col5 = st.columns([0.80, 0.9, 1.6, 1.6, 1.5])
# Copy code button
if col1.button("Instruct"):
    try:
        answer, code_exec = run_query(code_input)
    except Exception as e:
        st.error(str(e))

if col2.button("Annotate"):
    try:
        answer, code_exec = run_query(code_input, 'annotate')
    except Exception as e:
        st.error(str(e))

if col3.button("Translate to VoltScript"):
    try:
        answer, code_exec = run_query(code_input, 'translate', 'voltscript')
    except Exception as e:
        st.error(str(e))

if col4.button("Translate to JavaScript"):
    try:
        answer, code_exec = run_query(code_input, 'translate', 'javascript')
    except Exception as e:
        st.error(str(e))

if col5.button("Translate to Python"):
    try:
        answer, code_exec = run_query(code_input, 'translate', 'python')
    except Exception as e:
        st.error(str(e))

st.markdown(answer)
st.write(code_exec)

