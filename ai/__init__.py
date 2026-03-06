from .actor import ConvLSTM, HIDDEN_SIZE, NUM_ACTIONS_DISCRETE, NUM_COMM
from .critic import TeamPoolingCritic
from .comm import CommHandler

# 向後相容：舊程式碼 from ai import ConvSNN 仍然有效
ConvSNN = ConvLSTM
