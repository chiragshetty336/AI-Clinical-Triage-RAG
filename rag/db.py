import psycopg2


def get_connection():

    conn = psycopg2.connect(
        host="postgres", database="airflow", user="airflow", password="airflow"
    )

    return conn
