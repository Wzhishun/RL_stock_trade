from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from Callback.BestTestModelCallback import MyTestCallback
import wandb
import numpy as np
from DensePolicy import CustomPolicy
from StockEnv import StockEnv
# todo： 不同股票
ran = 3  # 0--17
policy_type_number = 0  # 政策网络选择 0，1，2，3,4
n_eval_episodes = 5  # 每次验证要跑多少个episode
eval_freq = 1024 * 10  # 验证频率，每训练多少step验证一次
test = "test1"  # 测试序号，test1,2,3,4...
seed_num = 0  # 随机数种子，0,1,2...
policy_type_list = ["DensePolicy"]
stock_list =['1332 JT', '1333 JT', '1334 JT', '1605 JT', '1721 JT', '1801 JT', '1802 JT', '1803 JT',
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
np.random.seed(seed_num)
wandb.init(project="Test1106", entity="prcwzs", sync_tensorboard=True,
           resume="PPO" + stock_list[ran] + test + "_seed" + str(seed_num))
wandb.log({"RLAlgo": "PPO",
           "reward_type": "extra profit",
           "stock": stock_list[ran],
           "policy": policy_type_list[policy_type_number]})
env = StockEnv(type='train', ran=ran)
env = DummyVecEnv([lambda: env])
test_env = StockEnv(type='test', ran=ran)
test_env = DummyVecEnv([lambda: test_env])
checkPointCallback = CheckpointCallback(save_freq=20480,
                                        save_path='./ModelSave/' + policy_type_list[policy_type_number] + "/",
                                        name_prefix="PPO2" + stock_list[ran] + "_" + policy_type_list[
                                            policy_type_number] + "_" + test + "_seed" + str(seed_num))
testCallback = MyTestCallback(test_env,
                              best_model_save_path='./ModelSave/' + policy_type_list[policy_type_number] + "/",
                              n_eval_episodes=n_eval_episodes,
                              eval_freq=eval_freq,
                              best_model_name="PPO2" + stock_list[ran] + "_" + policy_type_list[
                                  policy_type_number] + "_" + test + "_seed" + str(seed_num) + "_test_best_model")

model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log="./log/",n_steps=1024, seed=seed_num,n_cpu_tf_sess=1)
model.learn(total_timesteps=500000, callback=[checkPointCallback, testCallback])
