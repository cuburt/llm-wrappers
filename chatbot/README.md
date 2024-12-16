# Retrieval Augmented Chatbots
### To run server locally:
- go to chatbot/chatbot/scripts/server folder
- run pip install -r requirements
- run python server.py 
- or run exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 server:server 
- you can now access the following routes in localhost:8080 once server initialization is finished:
  - /models/palm:generate
  - /models/gemini:generate
  - /models/palm:generate-code
  - /models/gemini:generate-code

### To run streamlit app locally:
- go to chatbot/streamlit-app folder
- build docker locally and run on localhost, or
- run streamlit run iq.py --server.port=8081 --server.address=12 7.0.0.1
- you can now access the app in localhost:8081 once streamlit app is built.

### Input payload structure:

- :generate

        {
            "enable_rag": true,
            "query": "How to get started with Iris?"
        }
- :generate-code

        {
            "enable_rag": false,
            "query": "Translate this python code to javascript: print('hello world')"
        }
