import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

def main() -> None:
    print("Load data:")
    penguins = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
    print(penguins.shape)
    print()

    print("preprocess:")
    penguins.dropna(inplace=True)
    data = penguins[[
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]].values
    print(data.shape)
    print()

    print("vectorize:")
    X = StandardScaler().fit_transform(data)
    print(X.shape)
    print()

    print("Use umap:")
    model = UMAP()
    embeddings = model.fit_transform(X)
    print(embeddings.shape)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
