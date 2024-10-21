from pathlib import Path

import pandas as pd
from schemas import (
    Document,
    SampleConfig,
    add_prices
)


def main() -> None:
    filepath = Path("./data")
    config = SampleConfig.load(filepath)
    print(config)
    print()

    df = pd.DataFrame({
        "doc_id": [1, 2, 3],
        "title": list("abc"),
    })
    print(df)
    print()

    price_df = add_prices(df)
    print(price_df)
    print()

    records = price_df.to_dict(orient="records")
    doc = Document(**records[0])
    print(doc)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
