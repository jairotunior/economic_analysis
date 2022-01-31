from abc import abstractmethod, ABC


class Transformation(ABC):

    def __init__(self, **kwargs):
        assert kwargs.get('name', None), "Se debe definir un nombre."

        self.name = kwargs.get('name')

    @abstractmethod
    def transform(self):
        pass