import pandas as pd

from .dataset_information import extract_dataset_id, extract_host


def download_dataset(url):
    """
    Download the dataset from the given URL and return a DataFrame.

    Arguments:
        url: The URL of the dataset page.

    Returns:
        DataFrame with dataset information.
    """
    dataset_id = extract_dataset_id(url)
    host = extract_host(url)

    download_url = f"https://{host}/resource/{dataset_id}.csv"
    offset = 0
    limit = 5000
    dataset = []

    while True:
        csv_url = f"{download_url}?$limit={limit}&$offset={offset}"

        try:
            data = pd.read_csv(csv_url, low_memory=False)
        except Exception:
            break

        if data.empty:
            break

        dataset.append(data)
        offset += limit

    if dataset:
        df = pd.concat(dataset, ignore_index=True)
        del dataset
        return df

    return pd.DataFrame()
