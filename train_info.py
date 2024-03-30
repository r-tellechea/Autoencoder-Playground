from dataclasses import dataclass
import torch

@dataclass
class TrainInfo:
	trained: bool=False
	encoder: torch.nn.Module | None = None
	decoder: torch.nn.Module | None = None
	autocoder: torch.nn.Module | None = None
	loss_values: list[float] | None = None
	epochs: int=500
