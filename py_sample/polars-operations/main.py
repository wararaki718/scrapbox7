import polars as pl


def main() -> None:
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
    })
    print(df.shape)
    
    for id_, name in zip(df["id"], df["name"]):
        print(id_, name)
    print()

    print(df.select(pl.col("ide")))
    print("DONE")


if __name__ == "__main__":
    main()
