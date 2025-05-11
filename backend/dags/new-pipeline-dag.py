# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# import sys
# import os



# # Explicitly add both paths
# sys.path.insert(0, '/opt/airflow')
# sys.path.insert(0, '/opt/airflow/scripts')

# try:
#     from scripts.data.collect_data import collect_news_data
#     from scripts.data.preprocess_data import preprocess_data
#     from scripts.model.train_model import train_model
#     from scripts.model.test_model import test_model
#     print("Successfully imported all modules")
# except ImportError as e:
#     print(f"Import error: {e}")
#     # List directories for debugging
#     print(f"Files in /opt/airflow: {os.listdir('/opt/airflow')}")
#     if os.path.exists('/opt/airflow/scripts'):
#         print(f"Files in scripts dir: {os.listdir('/opt/airflow/scripts')}")
#     raise

# # Define default arguments
# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# # Create the DAG
# dag = DAG(
#     'news_data_pipeline',
#     default_args=default_args,
#     description='A pipeline for collecting, processing, and modeling news data',
#     schedule=timedelta(days=1),
#     start_date=datetime(2025, 5, 1),
#     catchup=False,
#     tags=['news', 'data', 'pipeline'],
# )

# # Task 1: Collect news data
# collect_data_task = PythonOperator(
#     task_id='collect_news_data',
#     python_callable=collect_news_data,
#     dag=dag,
# )

# # Task 2: Preprocess data
# preprocess_data_task = PythonOperator(
#     task_id='preprocess_data',
#     python_callable=preprocess_data,
#     dag=dag,
# )

# # Task 3: Train model
# train_model_task = PythonOperator(
#     task_id='train_model',
#     python_callable=train_model,
#     dag=dag,
# )

# # Task 4: Test model
# test_model_task = PythonOperator(
#     task_id='test_model',
#     python_callable=test_model,
#     dag=dag,
# )


# # Define task dependencies 
# collect_data_task >> preprocess_data_task >> train_model_task >> test_model_task


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
    start_date=datetime(2025, 5, 1),
    catchup=False,
    tags=['news', 'data', 'pipeline'],
)

# Define tasks with function references - don't import modules at top level
def collect_news_data_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/scripts')
    from scripts.data.collect_data import collect_news_data
    return collect_news_data()

def preprocess_data_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/scripts')
    from scripts.data.preprocess_data import preprocess_data
    return preprocess_data()

def train_model_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/scripts')
    from scripts.model.train_model import train_model
    return train_model()

def test_model_wrapper(**kwargs):
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/scripts')
    from scripts.model.test_model import test_model
    return test_model()

# Tasks
collect_data_task = PythonOperator(
    task_id='collect_news_data',
    python_callable=collect_news_data_wrapper,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_wrapper,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_wrapper,
    dag=dag,
)

test_model_task = PythonOperator(
    task_id='test_model',
    python_callable=test_model_wrapper,
    dag=dag,
)

# Dependencies
collect_data_task >> preprocess_data_task >> train_model_task >> test_model_task