from uuid import uuid4

from pydantic import (
    AnyUrl,
    BaseModel,
    EmailStr,
    Field,
    PositiveInt,
    UUID4,
    ValidationError,
)


def show_error(errors: list[dict]) -> None:
    for key, value in errors[0].items():
        print(f"  - {key:>5}: {value}")
    print()


class Item(BaseModel):
    item_id: UUID4 = Field(default_factory=uuid4)
    price: PositiveInt
    email: EmailStr
    url: AnyUrl


def main() -> None:
    # ok
    print("# safe")
    item = Item(
        item_id=uuid4(),
        price=100,
        email="test@sample.com",
        url="http://sample.com"
    )
    print(item)
    print()

    try:
        print("# positveInt check")
        item = Item(
            price=-100,
            email="test@sample.com",
            url="http://sample.com"
        )
    except ValidationError as e:
        show_error(e.errors())

    try:
        print("# email check")
        item = Item(
            price=100,
            email="test-sample-com",
            url="http://sample.com"
        )
    except ValidationError as e:
        show_error(e.errors())

    try:
        print("# url check")
        item = Item(
            price=100,
            email="test@sample.com",
            url="http-sample-com"
        )
    except ValidationError as e:
        show_error(e.errors())

    print("DONE")


if __name__ == "__main__":
    main()
