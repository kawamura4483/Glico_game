# ライブラリ類の読み込み
import numpy as np
import time
import copy
from math import inf
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import importlib
import gc

class Environment:
    def __init__(self, goal_reward, mul_win, mul_lose):
        self.goal_reward = goal_reward
        self.mul_win = mul_win
        self.mul_lose = mul_lose
        
    def play(self, state1, state2, pi1, pi2):
        # get hand1 and hand2
        while True:
            # hand1
            hands1 = list(pi1.keys())
            probs1 = list(pi1.values())
            hand1 = np.random.choice(hands1, p=probs1)
            # hand2
            hands2 = list(pi2.keys())
            probs2 = list(pi2.values())
            hand2 = np.random.choice(hands2, p=probs2)

            if hand1 != hand2:
                break
        
        # initial values
        next_state1, next_state2 = state1, state2
        done = False
                        
        # get game result
        if hand1 == "r":
            if hand2 == "s":
                a = 3
                next_state1 = state1 - a
                reward1, reward2 = self.mul_win*a, self.mul_lose*a
            if hand2 == "p":
                a = 6
                next_state2 = state2 - a
                reward1, reward2 = self.mul_lose*a, self.mul_win*a
        if hand1 == "s":
            if hand2 == "r":
                a = 3
                next_state2 = state2 - a
                reward1, reward2 = self.mul_lose*a, self.mul_win*a
            if hand2 == "p":
                a = 5
                next_state1 = state1 - a
                reward1, reward2 = self.mul_win*a, self.mul_lose*a
        if hand1 == "p":
            if hand2 == "r":
                a = 6
                next_state1 = state1 - a
                reward1, reward2 = self.mul_win*a, self.mul_lose*a
            if hand2 == "s":
                a = 5
                next_state2 = state2 - a
                reward1, reward2 = self.mul_lose*a, self.mul_win*a
        
        # if goaled
        if next_state1 <= 0:
            reward1 = state1 + self.goal_reward
            reward2 = self.mul_lose * self.goal_reward # 等倍
            done = True
        if next_state2 <= 0:
            reward2 = state2 + self.goal_reward
            reward1 = self.mul_lose * self.goal_reward
            done = True
        
        return next_state1, next_state2, hand1, hand2, reward1, reward2, done


class Agent:
    def __init__(self, ans, gamma, alpha, n, m):
        self.ans = ans
        self.gamma = gamma
        self.alpha = alpha
        self.n = n
        self.m = m
        self.state1 = n
        self.state2 = m
        
        random_hands = {'r':1/3, 's':1/3, 'p':1/3}
        self.pi = [[random_hands for i in range(self.m+1)] for j in range(self.n+1)]
        self.Q = [[{'r':0, 's':0, 'p':0} for _ in range(self.m+1)] for _ in range(self.n+1)] # バッチごとのQ値を持っておくリスト
        self.memory = []
    
    # 現在の点における確率分布を返す
    def get_pi(self):
        return self.pi[self.state1][self.state2]
    
    # memoryに必要な値を格納する
    def add(self, hand, reward):
        data = (self.state1, self.state2, hand, reward)
        self.memory.append(data)
    
    def update_position(self, next_state1, next_state2):
        self.state1 = next_state1
        self.state2 = next_state2

    # Qの計算
    def culculate_Q(self):
        G = 0
        for data in reversed(self.memory):
            state1, state2, hand, reward = data
            G = self.gamma * G + reward
            self.Q[state1][state2][hand] += (G - self.Q[state1][state2][hand]) * self.alpha
    
    # 最適方策のアップデート（Qを使う手法）soft-max関数
    def update_pi_softmax(self, tau, i, j, *Q): # Qの値を外部からとれるように
        if len(Q) == 0:
            Q = self.Q # 空の場合は、自身のQを使用する
        else:
            Q = Q[0] # Qに入力がある場合は、Qの中身を取り出す
        
        # (i, j)の地点のpiをQ値に基づいてupdate
        sum_exp_Qs = sum([np.exp(q/tau) for q in Q[i][j].values()]) # 分母を計算
        self.pi[i][j] = {'r': np.exp(Q[i][j]['r']/tau)/sum_exp_Qs, 's': np.exp(Q[i][j]['s']/tau)/sum_exp_Qs, 'p': np.exp(Q[i][j]['p']/tau)/sum_exp_Qs} # 確率分布を計算
    
    # ナッシュ均衡の回答をコピーする
    def copy_ans(self, n_, m_):
        for i in range(1, n_+1):
            for j in range(1, m_+1):
                self.pi[i][j] = self.ans[i][j]
    
    # 手の確率分布を与えられた確率分布に合わせる
    def synchronize(self, given_pi):
        self.pi = copy.deepcopy(given_pi)
    
    # エピソード終了後に、それぞれの値を初期値に直し、メモリーを消去する
    def episode_reset(self):
        self.state1 = self.n
        self.state2 = self.m
        self.memory.clear()
    
    def Q_reset(self):
        self.Q = [[{'r':0, 's':0, 'p':0} for _ in range(self.m+1)] for _ in range(self.n+1)]

    # 終わって良いかのチェックする関数
    def check(self, range_check):
        n = self.n
        m = self.m
        if (self.Q[n][m]['r'] < 0) or (self.Q[n][m]['s'] < 0) or (self.Q[n][m]['p'] < 0): return False
        # if not range_check: return True
        sorted_Q = sorted((self.Q[n][m]['r'], self.Q[n][m]['s'], self.Q[n][m]['p']), reverse=True) # 降順に並び替え
        max_Q = sorted_Q[0]
        second_Q = sorted_Q[1]
        min_Q = sorted_Q[2]
        if (max_Q < second_Q*1.5) and (min_Q > second_Q*0.5):
            print("Finish")
            return True
        return False
        
# ゲーム実施関数
def game(P1, P2, Env, episodes, diff, middle_quit=True):
    n = P1.n
    m = P1.m
    
    # Q値の最大値と最小値を計算し、変化が一定の値に収まった時点で終了させる
    min_r, min_s, min_p = inf, inf, inf
    max_r, max_s, max_p = -inf, -inf, -inf
    Q_dict = {'r':[], 's':[], 'p':[]}

    count = 0
    start_time = time.time()
    for _ in range(episodes):
        count += 1
        P1.episode_reset()
        P2.episode_reset()
        while True:
            state1 = P1.state1
            state2 = P2.state1
            pi1 = P1.get_pi()
            pi2 = P2.get_pi()

            next_state1, next_state2, hand1, hand2, reward1, reward2, done = Env.play(state1, state2, pi1, pi2)
            P1.add(hand1, reward1)
            P2.add(hand2, reward2)

            # 場所情報の更新
            P1.update_position(next_state1, next_state2)
            P2.update_position(next_state2, next_state1)

            if done:
                P1.culculate_Q()
                P2.culculate_Q()
                break
        
        # Q値を記憶
        Q_dict['r'].append(P1.Q[n][m]['r'])
        Q_dict['s'].append(P1.Q[n][m]['s'])
        Q_dict['p'].append(P1.Q[n][m]['p'])
        
        if middle_quit:
            # 過去10000回のQ値の最大値と最小値を記憶
            max_r = max(P1.Q[n][m]['r'], max_r)
            max_s = max(P1.Q[n][m]['s'], max_s)
            max_p = max(P1.Q[n][m]['p'], max_p)
            min_r = min(P1.Q[n][m]['r'], min_r)
            min_s = min(P1.Q[n][m]['s'], min_s)
            min_p = min(P1.Q[n][m]['p'], min_p)

            if count % 10000 == 0:
                if (max_r - min_r <= diff) and (max_s - min_s <= diff) and (max_p - min_p <= diff):
                    break
                else:
                    min_r, min_s, min_p = inf, inf, inf
                    max_r, max_s, max_p = -inf, -inf, -inf
    
    # 途中切り上げをした場合は、回数を表示
    if middle_quit: print("count:", count)
    end_time = time.time()
    print("time:", end_time - start_time)
    
    return Q_dict

def Q_show(Q_dict):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Q_dict['r'], label='Rock')
    ax.plot(Q_dict['s'], label='Scissors')
    ax.plot(Q_dict['p'], label='Paper')
    ax.legend()
    plt.show()
    
def Q_save(Q_dict, png_file_name):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Q_dict['r'], label='Rock')
    ax.plot(Q_dict['s'], label='Scissors')
    ax.plot(Q_dict['p'], label='Paper')
    ax.legend()
    plt.savefig(png_file_name)
    del fig, ax, Q_dict
    gc.collect()

def nash_Q(ans, gamma, alpha_nash, n, m, goal_reward, mul_win, mul_lose, episodes_nash, diff_nash, tau, middle_quit=False):
    # エージェントの構築
    P1 = Agent(ans, gamma, alpha_nash, n, m)
    P2 = Agent(ans, gamma, alpha_nash, m, n)
    
    # 環境の構築
    Env = Environment(goal_reward, mul_win, mul_lose)
    # n, mまでの確率分布の解答をナッシュ均衡からコピー
    P1.copy_ans(n, m)
    P2.copy_ans(m, n)

    nash_Q_dict = game(P1, P2, Env, episodes_nash, diff_nash, middle_quit)
    # Q_show(nash_Q_dict)
    
    return nash_Q_dict

def Learning(ans, gamma, alpha, n, m, goal_reward, mul_win, mul_lose, learning_number, episodes, diff, tau, middle_quit=True, range_check=False):
    # エージェントインスタンスの作成
    P1 = Agent(ans, gamma, alpha, n, m)
    P2 = Agent(ans, gamma, alpha, m, n)
    
    # n-1, m-1までの確率分布の解答をナッシュ均衡からコピー
    P1.copy_ans(n-1, m-1)
    P2.copy_ans(m-1, n-1)
    
    # 環境インスタンスの作成
    Env = Environment(goal_reward, mul_win, mul_lose)
    
    print("n, m: ", n, m)
    Q_dicts = [[]] # Q_dictを追加していくリスト
    
    for k in range(1, learning_number+1):
        print()
        print(k, "回目学習")
        print("学習前：", P1.pi[n][m])

        # 学習
        Q_dict = game(P1, P2, Env, episodes, diff, middle_quit)
        Q_dicts.append(Q_dict)

        # 学習結果の可視化
        Q_show(Q_dict)

        # 終了して良いかの確認
        if P1.check(range_check): # 幅によるチェックは行わない（全て正ならOK）
            break

        # softmaxによる値の更新
        P1.update_pi_softmax(tau, n, m)

        # P2の確率分布をP1に合わせる
        P2.synchronize(P1.pi)

        # Q値を初期化
        P1.Q_reset()
        P2.Q_reset()
    
    return Q_dicts