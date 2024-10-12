#!/bin/bash

while getopts i:p:m:n:t:o:k:p:s: flag
do
  case "${flag}" in
    i) article=${OPTARG};;
    p) PROJECT_ID=${OPTARG};;
    m) MODEL=${OPTARG};;
    n) prompt=${OPTARG};;
    t) TEMPERATURE=${OPTARG};;
    o) MAX_OUTPUT_TOKENS=${OPTARG};;
    k) TOP_K=${OPTARG};;
    p) TOP_P=${OPTARG};;
    s) service=${OPTARG};;
  esac
done
SENTIMENT_EXAMPLES="[{\"text\": \"I am absolutely thrilled with the excellent service and outstanding quality of the product!\",
                          \"label\": \"Highly Positive\"},
                          {\"text\": \"The team did an incredible job, and I couldn't be happier with the results.\",
                          \"label\": \"Highly Positive\"},
                          {\"text\": \"The event was well-organized, and I enjoyed every moment of it.\",
                          \"label\": \"Positive\"},
                          {\"text\": \"I had a great time at the event last night. The atmosphere was fantastic, and I got to meet some amazing people.\",
                          \"label\": \"Positive\"},
                          {\"text\": \"The weather today is neither too hot nor too cold. It's a typical day for this time of the year.\",
                          \"label\": \"Neutral\"},
                          {\"text\": \"The presentation covered various topics, providing an overview of the subject matter\",
                          \"label\": \"Neutral\"},
                          {\"text\": \"The customer support was unresponsive, and I had a hard time resolving my issue.\",
                          \"label\": \"Negative\"},
                          {\"text\": \"I was disappointed with the service at the restaurant. The food took forever to arrive, and the staff was unresponsive to our concerns.\",
                          \"label\": \"Negative\"},
                          {\"text\": \"The project was a complete failure, and we encountered numerous setbacks.\",
                          \"label\": \"Highly Negative\"},
                          {\"text\": \"The recent layoffs at our company have been devastating. Many of my colleagues, including myself, have lost our jobs, and it's been a very difficult time for everyone.\",
                          \"label\": \"Highly Negative\"}]"
TONE_EXAMPLES="[{\"text\": \"I felt a deep sense of sorrow and loss when I heard the news of my friend's passing.\",
                          \"label\": \"Sadness\"},
                          {\"text\": \"I felt heartbroken when I heard the news about the passing of my beloved pet.\",
                          \"label\": \"Sadness\"},
                          {\"text\": \"I was overjoyed and elated when I received the news of my promotion at work.\",
                          \"label\": \"Joy\"},
                          {\"text\": \"Winning the competition brought immense happiness and excitement to my life.\",
                          \"label\": \"Joy\"},
                          {\"text\": \"I was taken by surprise when my friends threw me a surprise birthday party.\",
                          \"label\": \"Surprise\"},
                          {\"text\": \"I was taken aback when I received an unexpected gift from a long-lost friend.\",
                          \"label\": \"Surprise\"},
                          {\"text\": \"The behavior of the rude customer at the store left me feeling disgusted.\",
                          \"label\": \"Disgust\"},
                          {\"text\": \"The sight and smell of the rotting garbage made me feel disgusted.\",
                          \"label\": \"Disgust\"},
                          {\"text\": \"The sudden loud noise startled me, and I felt a rush of fear.\",
                          \"label\": \"Fear\"},
                          {\"text\": \"Walking alone in the dark alley filled me with fear and anxiety.\",
                          \"label\": \"Fear\"},
                          {\"text\": \"The constant delays and poor service made me furious and angry with the airline.\",
                          \"label\": \"Anger\"},
                          {\"text\": \"I became furious when I discovered that someone had betrayed my trust.\",
                          \"label\": \"Anger\"}]"
case $MODEL in
  "mistral")
    GOOGLE_ACCESS_TOKEN="$(gcloud auth print-access-token)" || GOOGLE_ACCESS_TOKEN=`curl \
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
        -H "Metadata-Flavor: Google" | jq -r '.access_token'`
    case $service in
      "summariser")
      curl \
      -X POST \
      -H "Authorization: Bearer $GOOGLE_ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      https://us-east4-aiplatform.googleapis.com/v1/projects/303961749027/locations/us-east4/endpoints/8441460939831640064:predict -d \
      $"{
        \"instances\": [
          {
            \"prompt\": \"$prompt: $article\",
            \"max_tokens\": ${MAX_OUTPUT_TOKENS:-256}
          }
        ]
      }"
      ;;
      "sentiment-analysis")
      curl \
      -X POST \
      -H "Authorization: Bearer $GOOGLE_ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      https://us-east4-aiplatform.googleapis.com/v1/projects/303961749027/locations/us-east4/endpoints/8441460939831640064:predict -d \
      $"{
        \"instances\": [
          {
            \"prompt\": \"$prompt: $article \n\n $SENTIMENT_EXAMPLES\",
            \"max_tokens\": ${MAX_OUTPUT_TOKENS:-20}
          }
        ]
      }"
      ;;
      "emotion-analysis")
      curl \
      -X POST \
      -H "Authorization: Bearer $GOOGLE_ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      https://us-east4-aiplatform.googleapis.com/v1/projects/303961749027/locations/us-east4/endpoints/8441460939831640064:predict -d \
      $"{
        \"instances\": [
          {
            \"prompt\": \"$prompt: $article \n\n $TONE_EXAMPLES\",
            \"max_tokens\": ${MAX_OUTPUT_TOKENS:-10}
          }
        ]
      }"
      ;;
    esac
    ;;
  "palm")
    GOOGLE_ACCESS_TOKEN="$(gcloud auth print-access-token)" || GOOGLE_ACCESS_TOKEN=`curl \
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
        -H "Metadata-Flavor: Google" | jq -r '.access_token'`

    curl \
    -X POST \
    -H "Authorization: Bearer $GOOGLE_ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID:-hclsw-gcp-xai}/locations/us-central1/publishers/google/models/text-bison:predict -d \
    $"{
      \"instances\": [
        {
          \"content\": \"$prompt: $article\"
        }
      ],
      \"parameters\": {
        \"temperature\": ${TEMPERATURE:-0.2},
        \"maxOutputTokens\": ${MAX_OUTPUT_TOKENS:-256},
        \"topK\": ${TOP_K:-40},
        \"topP\": ${TOP_P:-0.95}
      }
    }"
    ;;
  "gemini")
    GOOGLE_ACCESS_TOKEN="$(gcloud auth print-access-token)" || GOOGLE_ACCESS_TOKEN=`curl \
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
        -H "Metadata-Flavor: Google" | jq -r '.access_token'`

    curl \
    -X POST \
    -H "Authorization: Bearer $GOOGLE_ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID:-hclsw-gcp-xai}/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent -d \
    $"{
      \"contents\": [
        {
          \"role\": \"user\",
          \"parts\": [
            {
                \"text\": \"$prompt: $article\"
            }
          ]
        }
      ],
      \"generation_config\": {
        \"maxOutputTokens\": ${MAX_OUTPUT_TOKENS:-256},
        \"temperature\": ${TEMPERATURE:-0.2},
        \"topP\": ${TOP_P:-0.95}
      }
    }"
    ;;
  "gpt")
    OPENAI_API_KEY="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"

    curl \
    -X POST \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    https://api.openai.com/v1/completions -d \
    $"{
      \"model\": \"text-davinci-003\",
      \"prompt\": \"$prompt: $article\",
      \"max_tokens\": ${MAX_OUTPUT_TOKENS:-256},
      \"temperature\": ${TEMPERATURE:-0.2},
      \"top_p\": ${TOP_P:-0.95}
    }"
    ;;
  "cohere")
    COHERE_API_KEY="yzAi16T727F83eMg75mWDgevEd7NVBDm6KTpfFhQ"

    case $service in
    "summariser")
      curl --request POST \
       --url https://api.cohere.ai/v1/summarize \
       --header "accept: application/json" \
       --header "authorization: Bearer $COHERE_API_KEY" \
       --header "content-type: application/json" \
       --data \
       $"{\"text\": \"$article\"}"
       ;;
    "sentiment-analysis")
      curl --request POST \
       --url https://api.cohere.ai/v1/classify \
       --header "accept: application/json" \
       --header "authorization: Bearer $COHERE_API_KEY" \
       --header "content-type: application/json" \
       --data \
       $"{\"inputs\": [\"$article\"],
          \"examples\": $SENTIMENT_EXAMPLES}"
       ;;
    "emotion-analysis")
      curl --request POST \
       --url https://api.cohere.ai/v1/classify \
       --header "accept: application/json" \
       --header "authorization: Bearer $COHERE_API_KEY" \
       --header "content-type: application/json" \
       --data \
       $"{\"inputs\": [\"$article\"],
          \"examples\": $TONE_EXAMPLES}"
       ;;
    "keyphrase-extraction")
      curl --request POST \
       --url https://api.cohere.ai/v1/generate \
       --header "accept: application/json" \
       --header "authorization: Bearer $COHERE_API_KEY" \
       --header "content-type: application/json" \
       --data \
       $"{\"prompt\": \"Extract 5 keywords from the following article: $article\",
          \"max_tokens\": ${MAX_OUTPUT_TOKENS:-256},
          \"temperature\": ${TEMPERATURE:-0.0},
          \"k\": 0,
          \"stop_sequences\": [],
          \"return_likelihoods\": \"NONE\"
          }"
       ;;
    esac
    ;;
esac
