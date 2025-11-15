import pathlib
import requests
import zipfile

def download_data(config):
    data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip'
    dataset_folder_path = pathlib.Path(config.dataset_folder)
    dataset_folder_path.mkdir(exist_ok=True)
    zip_data_path = dataset_folder_path / 'fra-eng.zip'
    if not zip_data_path.exists():
        response = requests.get(data_url)
        zip_data_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_data_path, "r") as zip_ref:
        zip_ref.extractall(dataset_folder_path)
