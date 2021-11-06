import gym
from gym import spaces
import numpy as np
import random
import pandas as pd
import pickle
import time
from sklearn import preprocessing
import wandb
import math


class StockEnv(gym.Env):
    def __init__(self, INITIAL_ACCOUNT_BALANCE=1000000, type='train', train_size=240, ran=0):
        self.ran = ran
        super(StockEnv, self).__init__()
        self.type = type
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.train_size = train_size
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10, 2),
                                            dtype=np.float32)
        self.stock_list = ['1332 JT', '1333 JT', '1334 JT', '1605 JT', '1721 JT', '1801 JT', '1802 JT', '1803 JT',
                           '1808 JT', '1812 JT', '1925 JT', '1928 JT', '1963 JT', '2002 JT', '2269 JT', '2282 JT',
                           '2413 JT', '2432 JT', '2501 JT', '2502 JT', '2503 JT', '2531 JT', '2768 JT', '2801 JT',
                           '2802 JT', '2871 JT', '2914 JT', '3086 JT', '3099 JT', '3101 JT', '3103 JT', '3105 JT',
                           '3110 JT', '3289 JT', '3382 JT', '3401 JT', '3402 JT', '3405 JT', '3407 JT', '3436 JT',
                           '3659 JT', '3861 JT', '3863 JT', '3864 JT', '3865 JT', '3893 JT', '4004 JT', '4005 JT',
                           '4021 JT', '4041 JT', '4042 JT', '4043 JT', '4061 JT', '4063 JT', '4151 JT', '4183 JT',
                           '4188 JT', '4208 JT', '4272 JT', '4324 JT', '4452 JT', '4502 JT', '4503 JT', '4506 JT',
                           '4507 JT', '4519 JT', '4523 JT', '4543 JT', '4568 JT', '4578 JT', '4631 JT', '4689 JT',
                           '4704 JT', '4751 JT', '4755 JT', '4901 JT', '4902 JT', '4911 JT', '5002 JT', '5019 JT',
                           '5020 JT', '5101 JT', '5108 JT', '5201 JT', '5202 JT', '5214 JT', '5232 JT', '5233 JT',
                           '5301 JT', '5332 JT', '5333 JT', '5401 JT', '5406 JT', '5411 JT', '5413 JT', '5541 JT',
                           '5631 JT', '5703 JT', '5706 JT', '5707 JT', '5711 JT', '5713 JT', '5714 JT', '5715 JT',
                           '5801 JT', '5802 JT', '5803 JT', '5901 JT', '6098 JT', '6103 JT', '6113 JT', '6178 JT',
                           '6301 JT', '6302 JT', '6305 JT', '6326 JT', '6361 JT', '6366 JT', '6367 JT', '6471 JT',
                           '6472 JT', '6473 JT', '6479 JT', '6501 JT', '6502 JT', '6503 JT', '6504 JT', '6506 JT',
                           '6508 JT', '6645 JT', '6674 JT', '6701 JT', '6702 JT', '6703 JT', '6724 JT', '6752 JT',
                           '6753 JT', '6758 JT', '6762 JT', '6767 JT', '6770 JT', '6773 JT', '6841 JT', '6857 JT',
                           '6902 JT', '6952 JT', '6954 JT', '6971 JT', '6976 JT', '6988 JT', '7003 JT', '7004 JT',
                           '7011 JT', '7012 JT', '7013 JT', '7186 JT', '7201 JT', '7202 JT', '7203 JT', '7205 JT',
                           '7211 JT', '7261 JT', '7267 JT', '7269 JT', '7270 JT', '7272 JT', '7731 JT', '7733 JT',
                           '7735 JT', '7751 JT', '7752 JT', '7762 JT', '7832 JT', '7911 JT', '7912 JT', '7951 JT',
                           '8001 JT', '8002 JT', '8015 JT', '8028 JT', '8031 JT', '8035 JT', '8053 JT', '8058 JT',
                           '8233 JT', '8252 JT', '8253 JT', '8267 JT', '8270 JT', '8303 JT', '8304 JT', '8306 JT',
                           '8308 JT', '8309 JT', '8316 JT', '8331 JT', '8332 JT', '8354 JT', '8355 JT', '8411 JT',
                           '8601 JT', '8604 JT', '8628 JT', '8630 JT', '8697 JT', '8725 JT', '8729 JT', '8750 JT',
                           '8766 JT', '8795 JT', '8801 JT', '8802 JT', '8803 JT', '8804 JT', '8815 JT', '8830 JT',
                           '9001 JT', '9005 JT', '9007 JT', '9008 JT', '9009 JT', '9020 JT', '9021 JT', '9022 JT',
                           '9062 JT', '9064 JT', '9101 JT', '9104 JT', '9107 JT', '9202 JT', '9301 JT', '9412 JT',
                           '9432 JT', '9433 JT', '9434 JT', '9437 JT', '9501 JT', '9502 JT', '9503 JT', '9531 JT',
                           '9532 JT', '9602 JT', '9613 JT', '9681 JT', '9735 JT', '9766 JT', '9983 JT', '9984 JT']

        self.stock_data = pd.read_csv("./data/" + self.stock_list[self.ran] + ".csv", index_col=0).iloc[:,1:]
        self.run_step = 0

    def _next_observation(self):
        array_1d = preprocessing.minmax_scale(
            np.array(self.stock_data.iloc[self.current_step - 9:self.current_step + 1, 0:2]),
            feature_range=(-1, 1), axis=0, copy=False)
        obs = array_1d
        return obs

    def reset(self):
        self.stock_choose = self.stock_list[self.ran]
        self.price_list = self.stock_data['last'].to_list()
        self.range1 = 10
        self.range2 = len(self.price_list) - 480 - 2
        self.range3 = len(self.price_list) - 240 - 2
        if self.type == 'train':
            self.current_step = random.randint(self.range1, self.range2)
        elif self.type == 'test':
            self.current_step = random.randint(self.range2, self.range3)

        self.total_asset = self.INITIAL_ACCOUNT_BALANCE
        self.stock_num = 0
        self.cash = self.INITIAL_ACCOUNT_BALANCE
        self.init_step = self.current_step
        self.init_price = self.price_list[self.current_step + 1]
        self.last_price = self.init_price
        self.this_price = self.init_price
        self.current_price = self.init_price
        self.done = False
        self.reward_sum = 0

        self.asset_rate_list = []
        self.price_rate_list = []

        return self._next_observation()

    def _take_action(self, action):
        self.last_asset = self.total_asset
        action_type = action[0]
        self.last_price = self.this_price
        self.current_price = self.price_list[self.current_step + 1]
        self.this_price = self.current_price
        self.total_asset = self.current_price * self.stock_num + self.cash
        if action_type > 0:
            buy_num = int(self.cash * action_type / (self.current_price * 1.00025))
            if buy_num * self.current_price >= 10000:
                self.cash = self.cash - buy_num * self.current_price * 1.00025
            elif buy_num * self.current_price < 10000 and buy_num > 0:
                self.cash = self.cash - buy_num * self.current_price - 5
            self.stock_num = self.stock_num + buy_num
            self.total_asset = self.cash + self.stock_num * self.current_price
        elif action_type <= 0:
            sell_num = int(self.stock_num * abs(action_type))
            if sell_num * self.current_price >= 10000:
                self.cash = self.cash + sell_num * self.current_price * (1 - 0.00125)
            elif sell_num * self.current_price < 10000 and sell_num > 0:
                self.cash = self.cash + sell_num * self.current_price * (1 - 0.001) - 5
            self.stock_num = self.stock_num - sell_num
            self.total_asset = self.cash + self.stock_num * self.current_price

        self.asset_rate_list.append(round(self.total_asset / self.last_asset - 1, 4))
        self.price_rate_list.append(round(self.current_price / self.last_price - 1, 4))

    def step(self, action):
        self.run_step = self.run_step + 1
        if self.current_step >= self.init_step + self.train_size or self.total_asset <= 0:
            # 计算贝塔系数
            def beta(list_stock, list_index):
                return (np.cov(list_stock, list_index))[0][1] / np.var(list_index)
            # 计算夏普比率
            def sharp_ratio(list, rf=0.02, days=240):
                a = pd.DataFrame(list) - rf / days
                return ((a.mean() * math.sqrt(days)) / a.std())[0]
            def max_drawdown(return_list):
                try:
                    index_j = np.argmax(np.maximum.accumulate(return_list) - return_list)  # 结束位置
                    index_i = np.argmax(return_list[:index_j])  # 开始位置
                    d = return_list[index_j] - return_list[index_i]  # 最大回撤
                except BaseException as e:
                    d = 0
                return d
            sharp_ratio = round(sharp_ratio(self.asset_rate_list), 4)
            beta = round(beta(self.asset_rate_list, self.price_rate_list), 4)
            max_drawdown = abs(round(max_drawdown(np.array(self.asset_rate_list)), 4))
            if self.type == 'train':
                wandb.log({'sharp ratio': sharp_ratio, "steps": self.run_step})
                wandb.log({'max drawdown': max_drawdown, "steps": self.run_step})
                wandb.log({'beta': beta, "steps": self.run_step})
                wandb.log({'extra profit rate': round(
                    self.total_asset / self.INITIAL_ACCOUNT_BALANCE - 1, 4) -
                                                round(self.current_price / self.init_price - 1, 4),
                           "steps": self.run_step})
            if self.type == "test":
                wandb.log({'sharp ratio test': sharp_ratio, "steps": self.run_step})
                wandb.log({'max drawdown test': max_drawdown, "steps": self.run_step})
                wandb.log({'beta test': beta, "steps": self.run_step})
                wandb.log({'extra profit rate test': round(
                    self.total_asset / self.INITIAL_ACCOUNT_BALANCE - 1, 4) -
                                                     round(self.current_price / self.init_price - 1, 4),
                           "steps": self.run_step})
            self.done = True
        self._take_action(action)
        self.current_step += 1
        reward = ((self.total_asset - self.last_asset) / self.last_asset -
                  (self.this_price - self.last_price) / self.last_price) * 100
        self.reward_sum = self.reward_sum + reward
        obs = self._next_observation()
        return obs, reward, self.done, {}

    def render(self, step):

        return 0
