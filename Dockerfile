FROM apache/airflow:2.7.3

USER airflow

RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    numpy \
    psycopg2-binary \
    pymupdf