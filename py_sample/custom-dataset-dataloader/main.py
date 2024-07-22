from pathlib import Path

from torch.utils.data import DataLoader

from utils import CSVGenerator
from dataset import CustomDataset


def main() -> None:
    csv_path = Path("./data")
    csv_generator = CSVGenerator()
    output_files = csv_generator.generate(csv_path)
    print(f"{csv_path}: {len(output_files)}")
    print()

    custom_dataset = CustomDataset(csv_path, is_shuffle=True)
    custom_loader = DataLoader(dataset=custom_dataset, batch_size=20)
    for ids, names in custom_loader:
        print(ids)
        print(names)
        print()

    print("DONE")


if __name__ == "__main__":
    main()
