# Workflow for observability demo on Miyabi

from typing import Annotated

import numpy as np
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

# NpStrict1DArrayF64 = Annotated[
#     np.ndarray[tuple[int,], np.dtype[np.float64]],
#     NpArrayPydanticAnnotation.factory(
#         data_type=np.float64, dimensions=1, strict_data_typing=True
#     ),
# ]

# NpStrict2DArrayF64 = Annotated[
#     np.ndarray[tuple[int, int], np.dtype[np.float64]],
#     NpArrayPydanticAnnotation.factory(
#         data_type=np.float64, dimensions=2, strict_data_typing=True
#     ),
# ]

# NpStrict4DArrayF64 = Annotated[
#     np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]],
#     NpArrayPydanticAnnotation.factory(
#         data_type=np.float64, dimensions=4, strict_data_typing=True
#     ),
# ]

NpStrict2DArrayBool = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.bool]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool, dimensions=2, strict_data_typing=True),
]

NpStrict1DArrayLL = Annotated[
    np.ndarray[tuple[int], np.dtype[np.longlong]],
    NpArrayPydanticAnnotation.factory(data_type=np.longlong, dimensions=1, strict_data_typing=True),
]
