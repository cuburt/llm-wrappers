# Code Interpreter

### To run inference pipeline locally:
- go to code_interpreter/server folder
- run pip install -r requirements
- run python server.py 
- or run uvicorn server:server --reload --port 8080 
- you can now access the following routes in localhost:8080 once server initialization is finished:
  - /copilot/models/codey:instruct
  - /copilot/models/codey:translate
  - /copilot/models/codey:annotate
  
### To build and run sandbox locally in a linux machine:
- go to code_interpreter/sandboxes folder
- run /bin/sh build_local.sh
- supported sandboxes are: python:3.10, javascript, and voltscript. Change the URL parameters accordingly
- to test, run curl -X POST -H "Content-Type: application/json" -d '{"query": "Print \"Hej!\""}' http://127.0.0.1:8081/sandboxes/voltscript

### To deploy dind conatiner (sandbox) in K8s (GKE):
- go to code_interpreter/sandboxes folder
- run /bin/sh deploy_k8s.sh

### Input payload structure for inference pipeline:

- :instruct

        {
            "query": "generate a voltscript code that prints \"hello world\"",
            "return_references": true
        }
- :translate

        {
            "target_lang": "voltscript",
            "query": "my_array = ['hi micheal', 'hi jim', 'hi pam', 'hi dwight']\ni = 0\nwhile i < len(my_array):\n    print(my_array[i])\n    i+=1",
            "return_references": true
        }
- :annotate

        {
            "query": "\nDim my_array(0 To 3) As Variant\nmy_array(0) = \"hi micheal\"\nmy_array(1) = \"hi jim\"\nmy_array(2) = \"hi pam\"\nmy_array(3) = \"hi dwight\"\nDim i As Integer\ni = 0\nWhile i < UBound(my_array)\n    Print my_array(i)\n    i++\nWend",
            "return_references": true
        }