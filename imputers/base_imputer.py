from abc import ABC, abstractmethod
import pandas as pd

class BaseImputer(ABC):
    @abstractmethod
    def impute(self, data: pd.DataFrame):
        pass
