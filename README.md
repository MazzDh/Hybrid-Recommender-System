# Hybrid Recommender System

A hybrid recommendation system combining FP-Growth association rules and Neural Collaborative Filtering (NCF) deep learning for Market Basket Analysis and product recommendations.

---

## Overview

This project provides a recommendation system designed for e-commerce datasets. It addresses Market Basket Analysis (frequently bought together items) and personalized user recommendations:

1. **FP-Growth Algorithm**: Generates association rules from item co-occurrence logs to recommend items frequently purchased together.
2. **Neural Collaborative Filtering (NCF)**: Uses deep learning embeddings to predict personalized user preferences.

---

## System Architecture

The application adopts a Client-Server micro-service architecture:

- **Backend (FastAPI)**: Serves RESTful API endpoints for item-based (`/recommend/by-item`) and user-based (`/recommend/by-user`) recommendations.
- **Frontend (Streamlit)**: Interactive web user interface for testing recommendations.
- **Data Pipeline (dbt & PostgreSQL)**: Transforms and seeds raw transaction datasets into structured relational analytical models.
- **Containerization (Docker)**: Docker Compose handles container orchestration for scalable deployment.

---

## Tech Stack

- **Programming Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Data Engineering**: dbt (Data Build Tool), PostgreSQL, Pandas, NumPy, mlxtend
- **Backend API**: FastAPI, Uvicorn
- **Frontend UI**: Streamlit
- **Containerization**: Docker, Docker Compose

---

## Project Structure

```text
Hybrid-Recommender-System/
├── backend/                # FastAPI backend endpoints and recommendation logic
├── dbt_project/            # dbt transformations, models, and seeds
├── ui/                     # Streamlit frontend user interface
├── data/                   # Datasets and generated association rules
├── models/                 # Pretrained NCF model weights
├── notebooks/              # Jupyter notebooks for model training & evaluation
├── docker/                 # Dockerfiles and Docker Compose configuration
└── requirements.txt        # Python dependencies
```

---

## Quick Start

### Method 1: Running with Docker (Recommended)

1. Navigate to the docker directory:
   ```bash
   cd docker
   ```

2. Build and start containers:
   ```bash
   docker compose up --build
   ```

3. Access services:
   - **Backend API Docs**: http://localhost:8000/docs
   - **Streamlit Web UI**: http://localhost:8501

### Method 2: Running Locally Without Docker

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run backend server:
   ```bash
   uvicorn backend.main:app --reload
   ```

3. Run frontend interface:
   ```bash
   cd ui
   streamlit run streamlit_app.py
   ```

---

## API Endpoints

- `GET /recommend/by-item?item=<item_name>&top_k=5`: Item-based recommendations via FP-Growth rules.
- `GET /recommend/by-user?user_id=<user_id>&top_k=5`: User-based recommendations via NCF model embeddings.

---

## License

Distributed under the MIT License. See LICENSE for more details.
