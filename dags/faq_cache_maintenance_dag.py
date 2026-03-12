from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import psycopg2


def clean_cache():

    conn = psycopg2.connect(
        host="airflow_postgres", database="airflow", user="airflow", password="airflow"
    )

    cur = conn.cursor()

    print("Cleaning duplicate cached queries...")

    cur.execute(
        """
        DELETE FROM faq_cache
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM faq_cache
            GROUP BY query
        );
    """
    )

    deleted = cur.rowcount

    conn.commit()
    cur.close()
    conn.close()

    print(f"Duplicate cache rows removed: {deleted}")


def cache_statistics():

    conn = psycopg2.connect(
        host="airflow_postgres", database="airflow", user="airflow", password="airflow"
    )

    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM faq_cache;")
    total = cur.fetchone()[0]

    print(f"Total cached queries: {total}")

    cur.execute(
        """
        SELECT triage_level, COUNT(*)
        FROM faq_cache
        GROUP BY triage_level
    """
    )

    rows = cur.fetchall()

    print("Triage distribution:")

    for level, count in rows:
        print(level, count)

    cur.close()
    conn.close()


with DAG(
    dag_id="faq_cache_maintenance",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    clean_task = PythonOperator(task_id="clean_cache", python_callable=clean_cache)

    stats_task = PythonOperator(
        task_id="cache_statistics", python_callable=cache_statistics
    )

    clean_task >> stats_task
