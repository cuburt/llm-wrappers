#!/bin/bash

while getopts t:c:i:p:m:e:l: flag
do
  case "${flag}" in
    t) text=${OPTARG};;
    c) context=${OPTARG};;
    i) BASE64_ENCODED_IMG=${OPTARG};;
    p) PROJECT_ID=${OPTARG};;
    m) mode=${OPTARG};;
    e) example=${OPTARG};;
    l) model=${OPTARG};;
  esac
done

API_ENDPOINT="us-central1-aiplatform.googleapis.com"
PROJECT_ID="hclsw-gcp-xai"
GOOGLE_ACCESS_TOKEN=`curl \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
    -H "Metadata-Flavor: Google" | jq -r '.access_token'`

case $mode in
    "embedding")
        if [ -n "$BASE64_ENCODED_IMG" ]; then
            curl -X POST \
              -H "Authorization: Bearer ${GOOGLE_ACCESS_TOKEN}" \
              -H "Content-Type: application/json; charset=utf-8" \
            "https://${API_ENDPOINT}/v1/projects/${PROJECT_ID:-hclsw-gcp-xai}/locations/us-central1/publishers/google/models/multimodalembedding@001:predict" \
             -d $"{ \"instances\": [
                    {
                      \"text\": \"${text}\",
                      \"image\": { \"bytesBase64Encoded\": \"${BASE64_ENCODED_IMG}\" }
                    }
                  ]
                  }"
        else
            curl -X POST \
              -H "Authorization: Bearer ${GOOGLE_ACCESS_TOKEN}" \
              -H "Content-Type: application/json; charset=utf-8" \
            "https://${API_ENDPOINT}/v1/projects/${PROJECT_ID:-hclsw-gcp-xai}/locations/us-central1/publishers/google/models/multimodalembedding@001:predict" \
             -d $"{ \"instances\": [
                    {
                      \"text\": \"${text}\"
                    }
                  ]
                  }"
        fi
    ;;
    "chat")
        case $model in
        "palm")
            MODEL_ID="text-bison"
            curl \
            -X POST \
            -H "Authorization: Bearer ${GOOGLE_ACCESS_TOKEN}" \
            -H "Content-Type: application/json" \
            "https://${API_ENDPOINT}/v1/projects/${PROJECT_ID:-hclsw-gcp-xai}/locations/us-central1/publishers/google/models/${MODEL_ID}:predict" \
            -d $"{
                \"instances\": [
                    {
                        \"content\": \"${text}\"
                    }
                ],
                \"parameters\": {
                    \"temperature\": 0.0,
                    \"maxOutputTokens\": 1024
                    }
                }"
        ;;
        "gemini")
            MODEL_ID="gemini-pro"
            curl \
            -X POST \
            -H "Authorization: Bearer $GOOGLE_ACCESS_TOKEN" \
            -H "Content-Type: application/json" \
            "https://${API_ENDPOINT}/v1/projects/${PROJECT_ID:-hclsw-gcp-xai}/locations/us-central1/publishers/google/models/${MODEL_ID}:streamGenerateContent" \
            -d $"{
                \"contents\": [
                    {
                        \"role\": \"user\",
                        \"parts\": [
                            {
                                \"text\": \"${text}\"
                            }
                        ]
                    }
                ],
                \"generation_config\": {
                    \"temperature\": 0.0,
                    \"maxOutputTokens\": 1024
                }
                }"
        ;;
        esac
    ;;
esac
