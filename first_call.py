
#=================Library====================
# 既存ライブラリ類の読み込み
import numpy as np
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import importlib

# 自作モジュールの読み込み
import sys
import os
sys.path.append(os.getcwd())

import module
importlib.reload(module)
from module import Environment
from module import Agent
from module import game
from module import Q_show
from module import Q_save
from module import nash_Q
from module import Learning



#=================Loading answer====================
# ナッシュ均衡の解答の読み込み
with open('ans.bin', 'rb') as p:
    ans = pickle.load(p)



#=================Parameter====================
# エージェント
gamma = 0.9
alpha = 10**(-5) # nash均衡の解答確認は別変数を用意
alpha_nash = 10**(-6)

# 環境
goal_reward = 0
mul_win = 1
mul_lose = -1

# 学習パラメータ
learning_number = 20 # 最大20回学習させる
episodes = 10**6 # 10^6
episodes_nash = 10**7 # 10^7
diff = 0.005
tau = 3.5






print("Complete")