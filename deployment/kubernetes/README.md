# Kubernetes Local Deployment for OHLCV Technical Analysis

![Minikube and Kubernetes](../../imgs/minikube_and_kubernetes.png)

This documentation covers the Kubernetes deployment of the **OHLCV Technical Analysis - Chart Pattern Recognition Service**, which predicts horizontal and ray support levels from chart images using trained deep learning models.

# Record of Deployment on Kubernetes

![Kubernetes Deploy](../../imgs/KubernetesDeploy.gif)

## 📋 Prerequisites

- [Docker](https://www.docker.com/) installed and running
- [minikube](https://minikube.sigs.k8s.io/docs/start/) installed
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- Python environment (for testing scripts and notebooks)

## 🚀 Deployment

### 1. Deploy to Local Kubernetes Cluster

```
# From the project root directory
./deployment/kubernetes/deploy.sh
```

The deployment script will:
- Start minikube
- Build the Docker image in minikube's environment
- Deploy the service to Kubernetes
- Wait for the pod to be ready
- Display service information
- Set up port forwarding to localhost:9696

### 2. Access the Service

```
# Get the service URL
minikube service ohlcv-prediction-service -n ohlcv-prediction --url

# Or open in browser directly
minikube service ohlcv-prediction-service -n ohlcv-prediction
```

### 3. View in Kubernetes Dashboard

```
# Open Kubernetes dashboard
minikube dashboard
```

**Important:** Select the **"ohlcv-prediction"** namespace to view your deployment.

## 🧪 Testing the Service

### Health Check

```
curl -X GET http://localhost:9696/health
```

**Response:**
```
{
  "service": "ohlcv-prediction",
  "status": "healthy",
  "timestamp": "2026-01-19 22:03:51"
}
```

### Image Prediction Endpoint

The models expect **input images** and output a **mask** highlighting:

- Horizontal support lines (`horizontal_unet.pth`)
- Ray support lines (`unet_ray.pth`)

To test predictions:

1. Convert your chart image to base64.
2. Send it in JSON to `/predict` endpoint.

Example JSON:

```
{
  "model": "horizontal",   # or "ray"
  "image": "<base64-encoded-image>"
}
```

**Response JSON:**
```
{
  "mask_base64": "<base64-encoded-mask-image>"
}
```

- `mask_base64` → PNG image mask with predicted support lines

### Test via Jupyter Notebook

```
# Navigate to deployment directory
cd deployment/flask

# Open testing notebook
jupyter notebook predict_test.ipynb
```

This notebook demonstrates sending an image to the API and visualizing the predicted mask.

## 📁 Deployment Structure

```
deployment/kubernetes/
├── deploy.sh              # Automated deployment script
├── deployment.yaml        # Kubernetes manifests (namespace, configmap, deployment, service)
└── README.md              # This file
```

## 🛠️ Useful Commands

```
# Check deployment status
kubectl get all -n ohlcv-prediction

# View pod logs
kubectl logs -l app=ohlcv-prediction -n ohlcv-prediction

# View detailed pod information
kubectl describe pod -l app=ohlcv-prediction -n ohlcv-prediction

# Delete deployment
kubectl delete -f deployment/kubernetes/deployment.yaml

# Stop minikube
minikube stop

# Start minikube again
minikube start

# Kill port forwarding
pkill -f "kubectl port-forward.*9696"
```

## 🔧 Service Configuration

- **Image:** `ohlcv-predictor:latest` (built locally in minikube)
- **Port:** Service runs on port 80, forwards to container port 9696
- **NodePort:** 30081 for external access
- **Namespace:** `ohlcv-prediction`
- **Health Checks:** Readiness and liveness probes on `/health`
- **Resources:**
  - CPU: 250m request, 500m limit
  - Memory: 512Mi request, 1Gi limit

## 📊 Architecture

The deployment creates:

1. **Namespace:** `ohlcv-prediction` for resource isolation
2. **ConfigMap:** Environment variables (e.g., model paths)
3. **Deployment:** Flask application container with auto-restart
4. **Service:** NodePort for external access
5. **Pod:** Runs the prediction service with the deep learning models

### Kubernetes Resource Flow

```
┌─────────────────────────────────────┐
│   Kubernetes Cluster (minikube)     │
├─────────────────────────────────────┤
│  Namespace: ohlcv-prediction        │
│                                     │
│  ┌─ Deployment ─────────────────┐   │
│  │ ohlcv-prediction-deployment  │   │
│  │ ┌─ Pod ──────────────────┐   │   │
│  │ │ Flask App (9696)       │   │   │
│  │ │ Deep Learning Models   │   │   │
│  │ │ Health Checks          │   │   │
│  │ └────────────────────────┘   │   │
│  └───────────────┬──────────────┘   │
│                  │                  │
│  ┌───────────────┴────────────────┐ │
│  │ Service: NodePort (30081)      │ │
│  │ Port Forwarding (9696)         │ │
│  └────────────────────────────────┘ │
└──────────> localhost:9696
```

## 🔄 Deployment Workflow

1. **Build Phase**
```
docker build -t ohlcv-predictor:latest -f deployment/flask/Dockerfile .
```

2. **Deploy Phase**
```
kubectl apply -f deployment/kubernetes/deployment.yaml
```

3. **Ready Phase**
- Pod ready when health check passes

4. **Port Forward Phase**
```
kubectl port-forward service/ohlcv-prediction-service 9696:80
```

5. **Access Phase**
- Service available at `http://localhost:9696`

## 🚨 Troubleshooting

- **Pod not running:** `kubectl get pods -n ohlcv-prediction`
- **View logs:** `kubectl logs -l app=ohlcv-prediction -n ohlcv-prediction`
- **Port forwarding issues:** `pkill -f "kubectl port-forward.*9696"`
- **Models not loading:** Ensure `.pth` files exist in `models/` before building

## 🧹 Cleanup

```
# Delete namespace and all resources
kubectl delete namespace ohlcv-prediction

# Stop minikube
minikube stop

# Delete minikube cluster
minikube delete
```