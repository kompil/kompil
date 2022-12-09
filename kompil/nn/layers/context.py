import torch


class ContextSave(torch.nn.Module):
    def __init__(self, varname: str, context: dict):
        super().__init__()

        self.__varname = varname
        self.__context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.__context[self.__varname] = x
        return x

    def extra_repr(self) -> str:
        return f'"{self.__varname}"'


class BuilderContextSave:
    TYPE = "context_save"

    @classmethod
    def elem(cls, varname: str) -> dict:
        return dict(type=cls.TYPE, varname=varname)

    @classmethod
    def make(cls, varname: int, context: dict, **kwargs) -> torch.nn.Module:
        return ContextSave(varname, context)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
