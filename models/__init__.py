from .model import regist_model, get_model
from .tsn import TSN

# regist models, sort by alphabet
regist_model("TSN", TSN)