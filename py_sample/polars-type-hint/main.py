import pandera.polars as pa
import polars as pl
from pandera.typing import DataFrame


class DataSchema(pa.DataFrameModel):
    a: int
    b: int
    c: str


@pa.check_types(inplace=True)
def get_data() -> DataFrame[DataSchema]:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["foo", "bar", "ham"],
        }
    )
    return df


def main() -> None:
    df = get_data()
    print(df)
    print("DONE")


if __name__ == "__main__":
    main()
