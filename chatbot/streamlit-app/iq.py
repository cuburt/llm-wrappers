import streamlit as st
import streamlit.components.v1 as components
import logging
import requests
import json
from google.cloud import bigquery
from streamlit_feedback import streamlit_feedback
import datetime
import base64


# Define your BigQuery dataset and table
project_id = "hclsw-gcp-xai"
dataset_id = "voltmx_chatbot"
feedback_tbl = "feedback_alpha"
session_tbl = "session_alpha"
history_tbl = "chat_history"
client = bigquery.Client(project=project_id)


# Function to insert data into BigQuery
def insert_data(rows_to_insert, type:str="feedback"):
    if type == "feedback":
        table = client.dataset(dataset_id).table(feedback_tbl)
    else:
        table = client.dataset(dataset_id).table(session_tbl)
    errors = client.insert_rows_json(table, rows_to_insert)
    if errors:
        st.toast(f"Failed to send data: {errors}")
    else:
        st.toast("Data sent successfully!")


def is_session_exist(session_id):
    query = f"""
            SELECT * FROM `{project_id}.{dataset_id}.{session_tbl}` where session_id = "{session_id}" LIMIT 1
            """
    query_job = client.query(query)
    results = query_job.result()
    return len([row for row in results]) != 0


def get_session_history(session_id):
    history_query = f"""
            SELECT * FROM `{project_id}.{dataset_id}.{history_tbl}` where session_id = "{session_id}" ORDER BY created_on ASC
            """
    history_query_job = client.query(history_query)
    history = history_query_job.result()
    if not history:
        history = []
    feedback_query = f"""
                    SELECT * FROM `{project_id}.{dataset_id}.{feedback_tbl}` where session_id = "{session_id}" ORDER BY time ASC
                    """
    feedback_query_job = client.query(feedback_query)
    feedbacks = list(feedback_query_job.result())
    for entry in history:
        st.session_state.messages.append({"id": entry["message_id"], "author": "user", "content": entry["message"]})
        fb = [f for f in feedbacks if (f["response"] == entry["response"] and f["query"] == entry["message"]) or (f["message_id"] and f["message_id"] == entry["message_id"])]
        if len(fb) == 0:
            st.session_state.messages.append({"author": "assistant", "content": entry["response"]})
        else:
            st.session_state.messages.append({"id": fb[-1]["message_id"],
                                              "author": "assistant",
                                              "content": entry["response"],
                                              "feedback": {"score": '👍' if fb[-1]["is_satisfactory"] else '👎',
                                                           "text": fb[-1]["comment"]}})


class Adapter:
    def post(self, payload, headers=None, endpoint=''):
        try:
            if headers:
                result = requests.post(self.url + endpoint, headers=headers, data=payload)
            else:
                result = requests.post(self.url + endpoint, data=payload)
            return result.json()
        except Exception:
            return {}

    def get(self, payload, headers, endpoint=''):
        try:
            result = requests.get(self.url + endpoint, headers=headers, data=payload)
            return result.json()
        except Exception:
            return {}


class IQAdapter(Adapter):
    def __init__(self):
        self.url = "https://voltmx-chatbot-server-alpha-mmbrfgm5va-uc.a.run.app/xairagllmiqbot/"

    def build_payload(self, enable_rag: bool = True, query: str = ""):
        payload = {"enable_rag": enable_rag, "query": query}
        return json.dumps(payload)


class CodeAdapter(Adapter):
    def __init__(self):
        self.url = "http://35.245.247.133/domino/copilot/models/"

    def build_payload(self, return_references: bool, query: str, task: str = 'instruct', target_lang: str = None):
        try:
            payload = {"return_references": return_references, "query": query}
            if task == 'translate':
                payload = {"return_references": return_references, "query": query, "target_lang": target_lang}
        except Exception as e:
            raise Exception(str(e))
        return json.dumps(payload)


class Agent:
    def __init__(self):
        self.agent_adapter = IQAdapter()

    def __call__(self, llm, query, enable_rag, session_id, enable_code_interpreter=False):
        payload = self.agent_adapter.build_payload(enable_rag, query)
        endpoint = f"get_{llm}_answer_generate?session_id={session_id}"
        res = self.agent_adapter.post(payload=payload, endpoint=endpoint)
        response = res['data']['prediction']
        return response.get('id'), response.get('answer'), response.get('source_documents')


class CodeAgent:
    def __init__(self):
        self.agent_adapter = CodeAdapter()

    def __call__(self, return_references, query, task='instruct', target_lang: str = '', llm: str = 'codey'):
        payload = self.agent_adapter.build_payload(return_references, query, task, target_lang)
        endpoint = f"{llm}:{task}"
        response = self.agent_adapter.post(payload=payload, endpoint=endpoint)
        return response.get('answer'), response.get('sandbox-output')


def _submit_feedback(user_response, result, emoji=None):
    st.toast(f"Feedback submitted: {user_response} \n Conversation: {result}", icon=emoji)
    # For display
    if len(st.session_state.messages) > 0:
        st.session_state.messages[-1]["feedback"] = user_response
    insert_data([{"time": datetime.datetime.utcnow().isoformat(),
                 "query": result["user"],
                 "response": result["assistant"],
                 "comment": user_response['text'],
                 "is_satisfactory": user_response['score'] == "👍",
                 "session_id": st.query_params.session_id,
                 "message_id": result["id"]}])
    return user_response.update(result)


def build_sources(sources):
    with res_placeholder:
        html = ""
        for r in sources:
            ctn = " ".join(r[0].split())[:512]
            ctn = ctn + f"... <a href=\"{r[1]}\" target=\"_blank\">See more.</a>"
            html += "<div class=\"row card\" style=\"border-radius: 3px; border-bottom: 1px solid #f2a9a2;margin-bottom: 10px; padding: 0px 10px 0px;\"><p style=\"font-family: Arial, sans-serif;\">" + ctn + "</p></div>"
        components.html(html, height=1000, scrolling=True)


def urlsafe_base64_encode(data: str) -> str:
    """
    Encodes a given string to URL-safe Base64.

    :param data: The string to encode.
    :return: The URL-safe Base64 encoded string.
    """
    # Convert the input string to bytes
    byte_data = data.encode('utf-8')

    # Encode the bytes to URL-safe Base64
    base64_bytes = base64.urlsafe_b64encode(byte_data)

    # Convert the Base64 bytes back to a string
    base64_string = base64_bytes.decode('utf-8')

    return base64_string


def _submit_session_form():
    session_id = urlsafe_base64_encode(f"{st.session_state.name}_{st.session_state.team}_{st.session_state.proficiency.lower()}")
    st.query_params.session_id = session_id
    if not is_session_exist(session_id):
        insert_data([{"created_on": datetime.datetime.utcnow().isoformat(),
                      "session_id": session_id,
                      "name": st.session_state.name,
                      "team": st.session_state.team,
                      "proficiency": st.session_state.proficiency}], type="session")


def is_query_disabled(yes=False):
    return yes or any(msg is None for msg in
        [m.get("feedback", None) for m in st.session_state.messages if m["author"] == "assistant"])


@st.cache_resource
def load_agents():
    agent = Agent()
    code_agent = CodeAgent()
    return agent, code_agent


if __name__ == "__main__":
    agent, code_agent = load_agents()
    models = ["palm", "gemini", "llama", "gpt"]
    logging.info("starting app...")
    st.title("VoltMX RAG")

    if "session_id" not in st.query_params or not is_session_exist(st.query_params.session_id):
        with st.form("session_form"):
            name = st.text_input("Name", key="name")
            team = st.text_input("Team", key="team")
            proficiency = st.selectbox("Product Proficiency", key="proficiency", options=("Beginner", "Intermediate", "Expert"))
            session_form_submit = st.form_submit_button("Begin", on_click=_submit_session_form)

    else:
        col1, col2, col3, col4 = st.columns(4)
        action1 = col1.button('Get started with Iris')
        action2 = col2.button('Invoke Foundry from within Iris')
        action3 = col3.button('Enable Internationalization')
        action4 = col4.button('Embed Cordova application')

        sidetab1, sidetab2 = st.sidebar.tabs(["Source Documents", "Settings"])

        with sidetab1:
            res_placeholder = st.empty()

        with sidetab2:
            llm = st.radio("Set LLM", models, index=1, horizontal=True)
            enable_rag = st.checkbox('Enable RAG', value=True)
            enable_code_interpreter = st.checkbox('Enable Code Interpreter', value=False)
            st.caption("Note: Enabling Code Interpreter will disable RAG and non-RAG agent.")

        if "sources" not in st.session_state:
            st.session_state.sources = []

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if len(st.session_state.messages) == 0:  # only retrieve on start up
            get_session_history(st.query_params.session_id)

        # Display chat messages from history on app rerun
        for n, message in enumerate(st.session_state.messages):
            with st.chat_message(message["author"]):
                st.markdown(message["content"])

            if message["author"] == "assistant" and n >= 0:
                feedback_key = f"feedback_{int(n / 2)}"

                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None

                if "feedback" not in message:
                    feedback = streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="Please provide extra information",
                        on_submit=_submit_feedback,
                        key=feedback_key,
                        kwargs={"result": {"id": st.session_state.messages[n-1]["id"], "user": st.session_state.messages[n-1]["content"], "assistant": message["content"]}}
                    )

            if "feedback" in message and message["feedback"]['text'] and message['feedback']['score'] == '👍':
                st.success(f"Comment: {message['feedback']['text']}")

            elif "feedback" in message and message["feedback"]['text'] and message['feedback']['score'] == '👎':
                st.error(f"Comment: {message['feedback']['text']}")

        build_sources(st.session_state.sources)

        # Accept user input
        res = []

        query = st.chat_input(disabled=is_query_disabled())

        if query or action1 or action2 or action3 or action4:

            if action1:
                query = "How to get started with Iris?"
            elif action2:
                query = "How to invoke Foundry from within Iris?"
            elif action3:
                query = "How to enable Internationalization?"
            elif action4:
                query = "How to embed a Cordova application?"

            # Add user message to chat history
            st.session_state.messages.append({"author": "user", "content": query.replace('\'','')})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(query)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                # (self, return_references, query, task='instruct', target_lang: str='', llm: str='codey'):
                if enable_rag:
                    message_id, response, res = agent(llm=llm, query=query, enable_rag=True, session_id=st.query_params.session_id)
                else:
                    message_id, response, res = agent(llm=llm, query=query, enable_rag=False, session_id=st.query_params.session_id)

                if enable_code_interpreter:
                    response, code_exec = code_agent(return_references=False, query=query)
                    message_id, response = None, response + f"\nOutput from sandbox runtime:\n{code_exec}"

                # Display q&a response
                message_placeholder.markdown(response if response else "")

            if res:
                st.session_state.sources = res
                # build_sources(res)

            # Update user's message
            st.session_state.messages[-1]["id"] = message_id

            # Append q&a response
            st.session_state.messages.append({"id": message_id, "author": "assistant", "content": response})
            st.rerun()
