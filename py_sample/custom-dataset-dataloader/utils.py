from typing import List
from pathlib import Path, PosixPath

import pandas as pd


class CSVGenerator:
    def generate(self, output_dir: Path, n_csv: int=10, n_data: int=10) -> List[PosixPath]:
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True, parents=True)
            print(f"create '{output_dir}'")

        for i in range(n_csv):
            data = [{"id": i*n_data + j, "name": f"name_{i}_{j}"} for j in range(n_data)]
            filename = output_dir / f"file_{i}.csv"
            pd.DataFrame(data).to_csv(filename, index=None)

        return sorted(output_dir.glob("*.csv"))
