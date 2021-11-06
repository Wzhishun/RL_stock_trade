# Test1106
Test1106
You can see the lab in wandb file. It is public.
https://wandb.ai/prcwzs/Test1106/workspace?workspace=user-prcwzs

Due to my poor computer power and the noise of the computer fan (I am in CUHK library and it is really annoy my neighbors ), I just train several stocks not all securities you provided.


I use PPO(Proximal Policy Optimization) + MLP（Multilayer Perceptron） to create a RL(Reinforcement Learning) agent to trade stock you provided.

I do not choose to use traditional indiactor to trade stock cause I do think it will lead to the result of strategy homogeneity. 

However, using machine learning methods will face black box problems, which may cause systemic risks.

StockEnv.py is a Chinese stock trading env where transaction fee is 0.125%. StockPlotEnv.py is major in plot the trading image. quant.py is in charge of data process. PPO_run1.py is tha main fuction of running RL process. Test.zip includes other train models and data. run_plot.py is a plot process.


packages:

pickle

tensorflow-1.15

stable-baselines

pickle

scikit-learn

gym

Indicator	B&H	DNN-PPO	B&H	DNN-PPO	B&H	DNN-PPO	B&H	DNN-PPO
Profit rate	-28.83%	6.55%	-30.94%	-14.71%	Data is less		-16.24%	15.38%
Beta	100.00%	17.27%	100.00%	-30.21%			100.00%	32.12%
Sharp ratio	-40.20%	53.14%	-58.20%	71.45%			-9.03%	65.87%
Max dropdown	16.28%	13.96%	16.29%	14.53%			22.66%	21.85%


![Uploading image.png…]()


![image](https://user-images.githubusercontent.com/49648647/140601769-ee73b35d-a663-40ba-b87b-e493f20e74c0.png)
