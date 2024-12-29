import os
from pathlib import Path
from urllib.parse import urlparse

import requests


def download_pdfs(download_dir: Path=Path("data")) -> list[Path]:
    urls = [
        "https://www.mhlw.go.jp/content/001299412.pdf",
        "https://www.mhlw.go.jp/content/001249560.pdf",
        "https://www.mhlw.go.jp/content/001242967.pdf",
    ]

    for url in urls:
        result = urlparse(url)
        download_path = download_dir / os.path.basename(result.path)
        if download_path.exists():
            print(f"'{download_path}' is always existed!")
            continue

        response = requests.get(url)
        with open(download_path, "wb") as f:
            f.write(response.content)
        print(f"save '{download_path}'")

    return list(download_dir.glob("*.pdf"))
