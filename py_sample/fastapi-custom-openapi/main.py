import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field


class Item(BaseModel):
    item_id: int = Field(format="int64")
    name: str
    price: float = Field(format="double")


app = FastAPI()


@app.get("/item", response_model=Item)
def get_item() -> Item:
    return Item(
        item_id=1,
        name="name",
        price=10.0
    )


def main() -> None:
    schema = app.openapi()
    with open("schema/openapi.yaml", "wt") as f:
        yaml.safe_dump(schema, f)
    print("DONE")


if __name__ == "__main__":
    main()
