from pathlib import Path

import requests


def download_pdf(download_path: Path = Path("download/sample.pdf")) -> Path:
    # if pdf exists, skip download
    if download_path.exists():
        return download_path
    
    # download
    url = "https://abc.xyz/assets/43/44/675b83d7455885c4615d848d52a4/goog-10-k-2023.pdf"
    response = requests.get(url)
    with open(download_path, "wb") as f:
        f.write(response.content)

    return download_path
