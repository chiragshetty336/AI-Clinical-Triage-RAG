from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from rag.pipeline import ingest_documents, update_faiss_index

default_args = {
    "owner": "medical_rag",
    "retries": 1,
}

with DAG(
    dag_id="medical_rag_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",  # runs daily
    catchup=False,
    default_args=default_args,
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_documents",
        python_callable=ingest_documents,
    )

    index_task = PythonOperator(
        task_id="update_faiss_index",
        python_callable=update_faiss_index,
    )

    ingest_task >> index_task
