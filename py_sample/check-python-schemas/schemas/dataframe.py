import pandera

from pandera import DataFrameModel
from pandera.typing import DataFrame, Series


class InputSchema(DataFrameModel):
    doc_id: Series[int]
    title: Series[str]


class OutputSchema(DataFrameModel):
    doc_id: Series[int]
    title: Series[str]
    price: Series[int]


@pandera.check_types
def add_prices(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    result_df = df.assign(price=[100 for _ in range(df.shape[0])])
    return result_df
