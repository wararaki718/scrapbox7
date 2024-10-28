import pandas as pd


def main() -> None:
    sr = pd.Series([False, True, float("nan")])
    print(sr)
    print()

    # warning
    print("# warning")
    print(sr.fillna(False))
    print()

    print("# disable warning")
    with pd.option_context("future.no_silent_downcasting", True):
        a = sr.fillna(False)
        print(a)
    print()

    print("# astype")
    a = sr.astype(bool).fillna(False)
    print(a)
    print()

    print("# warning: after astype")
    a = sr.fillna(False).astype(bool)
    print(a)
    print()

    print("# overwrite")
    sr[sr.isna()] = False
    sr = sr.astype(bool)
    print(sr)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
