from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'news_data_pipeline',
    default_args=default_args,
    description='A pipeline for collecting, processing, and modeling news data',
    schedule=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['news', 'data', 'pipeline'],
)

# Define wrapper functions to delay imports until runtime
def collect_news_data_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    from scripts.data.collect_data import collect_news_data
    return collect_news_data()

def preprocess_data_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    from scripts.data.preprocess_data import preprocess_data
    return preprocess_data()

def train_model_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    from scripts.model.train_model import train_model
    return train_model()

def test_model_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    from scripts.model.test_model import test_model
    return test_model()

# Task 1: Collect news data
collect_data_task = PythonOperator(
    task_id='collect_news_data',
    python_callable=collect_news_data_wrapper,
    dag=dag,
)

# Task 2: Preprocess data
preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_wrapper,
    dag=dag,
)

# Task 3: Train model
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_wrapper,
    dag=dag,
)

# Task 4: Test model
test_model_task = PythonOperator(
    task_id='test_model',
    python_callable=test_model_wrapper,
    dag=dag,
)

# Define task dependencies 
collect_data_task >> preprocess_data_task >> train_model_task >> test_model_task