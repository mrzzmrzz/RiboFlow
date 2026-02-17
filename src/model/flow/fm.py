from abc import ABC, abstractmethod
from typing import Any


class FM(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def interpolant(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def sampling(self, *args: Any, **kwargs: Any) -> Any:
        pass
