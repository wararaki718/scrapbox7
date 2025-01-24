from pandera import DataFrameModel


class User(DataFrameModel):
    user_id: int
    name: str
    age: int


class Item(DataFrameModel):
    item_id: int
    name: str
    price: float
