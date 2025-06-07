from datetime import datetime, date

from airflow import DAG
from airflow.operators.python import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

with DAG(
    dag_id = "hiring_lineal",
    description="A lineal dag",
    start_date = datetime(2024,10,1),
    catchup=False,
    default_args = {"date": str(date.today())},
    schedule=None
) as dag:
    
    marker_task = EmptyOperator(task_id="Starting_the_processd", retries =2)

    folder_task = PythonOperator(
        task_id="Creating_folders",
        python_callable = create_folders
    )

    download_dataset_task = BashOperator(
        task_id='Download_dataset',
        bash_command='curl -o ./{{ dag.default_args.date }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv'
    )

    holdout_task = PythonOperator(
        task_id="Holdout",
        python_callable= split_data
    )

    preprocess_and_train_task = PythonOperator(
        task_id="Preprocess_and_train",
        python_callable=preprocess_and_train
    )

    gradio_gui_task = PythonOperator(
        task_id="Gradio_GUI",
        python_callable=gradio_interface
    )

    # pipeline definition

    marker_task >> folder_task >> download_dataset_task >> holdout_task >> preprocess_and_train_task >> gradio_gui_task