import lightning

from model import LightningModel, MultiTaskModel
from utils import get_dataloader


def main() -> None:
    train_dataloader, test_dataloader = get_dataloader()
    print(len(train_dataloader))
    print(len(test_dataloader))
    print()

    model = LightningModel(model=MultiTaskModel(n_input=784))
    trainer = lightning.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader)
    print("train finished!")
    print()

    trainer.test(model, test_dataloader)
    print("test finished")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
