# DiaSense

DiaSense is an end-to-end machine learning platform designed to predict the risk of diabetes based on clinical metrics. The project segments patients into risk groups using unsupervised clustering (KMeans) and trains various classification models to predict these risk profiles from new clinical data. 

The entire pipeline is containerized and orchestrated to support automated retraining, experiment tracking, and live model serving.

## Project Architecture

The architecture consists of the following microservices orchestrated via Docker Compose:

* **MLflow Tracking Server**: Tracks model hyperparameters, metrics, and registers the best performing model in the Model Registry.
* **FastAPI Backend**: Serves the best registered model via a REST API.
* **Streamlit Application**: A web-based clinical user interface for medical professionals to input patient data and receive risk predictions.
* **Apache Airflow**: Orchestrates the automated retraining pipeline (extract, preprocess, cluster, train, and register).

## Project Structure

* `dags/`: Contains the Airflow Directed Acyclic Graphs (DAGs) for automated pipeline execution.
* `data/`: Storage for raw and preprocessed datasets.
* `docker/`: Dockerfiles for each individual microservice.
* `models/`: Local storage for serialized models (pipelines, KMeans, Scikit-learn estimators).
* `notebooks/`: Jupyter notebooks used for initial EDA, preprocessing, and clustering experimentation.
* `reports/`: Generated artifacts such as classification metrics and confusion matrices.
* `src/diasense/`: The core Python package containing the logic for data loading, preprocessing, clustering, classification, APIs, and the UI.

## Getting Started

### Prerequisites

Ensure you have Docker and Docker Compose installed on your local machine. 

### Running the Services

To build and start all services, run the following command from the root of the project:

```bash
docker compose up -d --build
```

Once the containers are healthy, the services will be available at:

* **Streamlit UI**: http://localhost:8501
* **FastAPI Swagger Docs**: http://localhost:8000/docs
* **MLflow UI**: http://localhost:5000
* **Airflow UI**: http://localhost:8080 

Note: The default username for Airflow is `admin`. The password will be generated automatically in the Airflow container logs upon the first run.

## Workflow

1. **Exploration**: Initial data analysis is available in the `notebooks/` directory.
2. **Training**: The core model training loop tests K-Nearest Neighbors, Support Vector Machines, Logistic Regression, and Random Forest. 
3. **Tracking**: The pipeline compares the models based on the F1-Score, logs the results to MLflow, and automatically promotes the highest performing model to the "Production" stage.
4. **Orchestration**: The Airflow DAG (`diasense_retraining_pipeline`) executes this entire workflow from preprocessing to MLflow registration on a weekly schedule.
5. **Serving**: The FastAPI application dynamically loads the "Production" model from MLflow and exposes a `/predict` endpoint.

## Development

The Python code is managed as a standard package. To develop locally outside of Docker, install the package in editable mode:

```bash
pip install -e ./src
```
