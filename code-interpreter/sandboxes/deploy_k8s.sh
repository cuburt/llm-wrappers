#!/bin/bash

# Define YAML content
read -r -d '' YAML_CONTENT <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sandbox

spec:
  replicas: 1
  selector:
    matchLabels:
      app: sandbox-app
  template:
    metadata:
      labels:
        app: sandbox-app

    spec:
      containers:
        - name: sandbox-app-container
          image: us-east4-docker.pkg.dev/hclsw-gcp-xai/img-repo/code-interpreter/sandbox:0.2
          ports:
            - containerPort: 8081
              protocol: TCP
          securityContext:
              privileged: true
EOF

# Save YAML content to a temporary file
echo "$YAML_CONTENT" > temp.yaml

# Apply YAML using kubectl
kubectl apply -f temp.yaml

# Clean up temporary file
rm temp.yaml

# Wait for the deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available deployment/sandbox

# Get the pod name from the deployment
POD_NAME=$(kubectl get pods -l app=sandbox-app -o jsonpath="{.items[0].metadata.name}")

kubectl exec $POD_NAME -- /bin/sh -c "cd /sandboxes/voltscript && /bin/sh build.sh"
sleep 30
kubectl exec $POD_NAME -- /bin/sh -c "cd /sandboxes/javascript && /bin/sh build.sh"
sleep 30
kubectl exec $POD_NAME -- /bin/sh -c "cd /sandboxes/python && /bin/sh build.sh"
sleep 30
kubectl exec $POD_NAME -- /bin/sh -c "cd /sandboxes/server && /bin/sh build.sh"
sleep 30
kubectl expose deployment -n default sandbox --type=LoadBalancer --port=80 --target-port=8081 --protocol=TCP --name=sandbox-http