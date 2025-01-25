from pandera import DataFrameModel


class User(DataFrameModel):
    user_id: int
    name: str
    age: int


class CustomUser(DataFrameModel):
    user_id: int
    name: str
    age: int
    custom: str


class Item(DataFrameModel):
    item_id: int
    name: str
    price: float
