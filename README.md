# sepsis-mlops


### Preprocessing

cd components/preprocess
docker build -t sepsis-preprocess:v1 .


docker run -v $(pwd)/data:/data sepsis-preprocess:v1 --input /data/raw.csv --output /data/clean.csv

python components/preprocess/preprocess.py --input data/sepsis/training_setA/training/p000001.psv --output data/clean-3.csv

## train
### Rebuild training image if you made changes to train.py
sudo docker build -t sepsis-train:v1 ./components/train

### Run training using the file we just verified
sudo docker run -v $(pwd)/data:/data sepsis-train:v1 \
  --input /data/clean-2.csv


## Setting Kubeflow

### kubectl 
$ curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
$ sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# k3d
$ wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

### k9s
sudo dnf install https://github.com/derailed/k9s/releases/download/v0.31.7/k9s_linux_amd64.rpm

### create cluster  
sudo k3d cluster create --k3s-arg "--disable=traefik@server:*" --registry-create personal-project-registry personal-project-cluster

### pushing images to k3d registry
Add an /etc/hosts entry to allow you to push images into the K3d registry
Add a entry to /etc/hosts that maps the hostname personal-project-registry to 127.0.0.1

For example, change this:
127.0.0.1 localhost localhost.localdomain localhost4 localhost4.localdomain4

To this:
127.0.0.1 localhost localhost.localdomain localhost4 localhost4.localdomain4 personal-project-registry

### Get the port exposed by the registry
$ sudo docker ps -f name=personal-project-registry --format "{{.Ports}}"
0.0.0.0:35594->5000/tcp

# Note that, in this case, the exposed port on the host is 35594

# Tag a local/custom image with this registry & port
$ sudo docker tag nginx:latest personal-project-registry:35594/nginx:latest

### Push the image to the registry
$ sudo docker push personal-project-registry:35594/nginx:latest


# Installing standalone kubeflow pipelines (lightweight version)

# Set the version
export PIPELINE_VERSION=2.2.0

# Apply the manifests
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

### Install the core components
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"

Ultimately i had to grab the image from : 

sudo docker tag ocdr/ml-pipeline-ui:dev gcr.io/ml-pipeline/frontend:2.0.5

