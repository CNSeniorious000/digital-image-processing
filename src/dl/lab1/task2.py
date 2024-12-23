import torch

from .task1 import ConvNet

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ==================================================================
# 2. 找出task1中判错的图像，并组成自己的学号
# ==================================================================

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("task1.ckpt")
model = ConvNet().to(device)
model.load_state_dict(state_dict=ckpt, strict=True)

# ==================================================================
# 请完成这一部分，找出判错的图像，并组成自己的学号


# ==================================================================
