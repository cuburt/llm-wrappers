// Go server for shell-script-as-a-service.
package main

import (
        "log"
        "net/http"
        "os"
        "os/exec"
        "encoding/json"
        "github.com/gorilla/mux"
)

func main() {

        // register functions to handle all requests
        r := mux.NewRouter()
        r.HandleFunc("/models", modelsHandler)
        r.HandleFunc("/models/{model}/tasks", tasksHandler)
        r.HandleFunc("/models/{model}/tasks/{task}:predict", predictHandler)
//         http.Handle("/", r)

        // use PORT environment variable, or default to 8080
        port := os.Getenv("PORT")
        if port == "" {
                port = "8080"
        }

        // start the web server on port and accept requests
        log.Printf("Server listening on port %s", port)
        log.Fatal(http.ListenAndServe(":"+port, r))

}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method == "GET"{
        w.WriteHeader(403)
    } else {
        w.WriteHeader(404)
    }
}

func tasksHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method == "GET"{
        w.WriteHeader(403)
    } else {
        w.WriteHeader(404)
    }
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
    params := mux.Vars(r)
    task := params["task"]
    model := params["model"]
    if r.Method == "GET"{
        w.WriteHeader(404)
    } else {
        // Set the return Content-Type as JSON like before
        w.Header().Set("Content-Type", "application/json")
        // Read http.Request body
        payload := make(map[string]interface{})
        err := json.NewDecoder(r.Body).Decode(&payload)
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }
        text := payload["text"].(string)
        prompt := ""
        switch task {
        case "summariser":
            prompt = "Provide a summary with about two sentences for the following article"
        case "sentiment-analysis":
            prompt = "Return only the sentiment. Classify the sentiment whether its 'Highly Negative', 'Negative', 'Neutral', 'Positive', or 'Highly Positive' of the following article"
        case "emotion-analysis":
            prompt = "Return only the tone. Classify the tone whether its 'Sadness', 'Joy', 'Surprise', 'Disgust', 'Fear', or 'Anger' of the following article"
        case "keyphrase-extraction":
            prompt = "Extract 5 keyphrases from the following article"
        }

        // Run the script.sh and input body.input
        cmd := exec.Command("/bin/bash", "script.sh", "-n", prompt,"-i", text, "-s", task, "-m", model)
        cmd.Stderr = os.Stderr
        out, err := cmd.Output()
        if err != nil {
                w.WriteHeader(500)
        }
        w.Write(out)
    }
}