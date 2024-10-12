#!/bin/bash

GOOGLE_ACCESS_TOKEN="$(gcloud auth print-access-token)" || GOOGLE_ACCESS_TOKEN=`curl \
"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
-H "Metadata-Flavor: Google" | jq -r '.access_token'`

echo -n "$GOOGLE_ACCESS_TOKEN"
