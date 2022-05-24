import os
import argparse
import DataManager
import utils
from Learner import DIRILearner

"""
상승
010140: 삼성중공업
006280: 녹십자
009830: 한화솔루션
011170: 롯데케미칼
010060: OCI
034220: LG디스플레이
000810: 삼성화재

박스권/하락
010140: 삼성중공업
013570: 디와이
010690: 화신
000910: 유니온
010060: OCI
034220: LG디스플레이
009540: 한국조선해양
053800: 안랩
"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--stock_code", nargs="+", default= ["010140", "000810", "034220"])
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--tau", type=float, default=0.005)
  parser.add_argument("--delta", type=float, default=0.005) #0.005
  parser.add_argument("--discount_factor", type=float, default=0.9)
  parser.add_argument("--num_episode", type=int, default=50)
  parser.add_argument("--balance", type=int, default=15000000)
  parser.add_argument("--batch_size", type=int, default=30)
  parser.add_argument("--memory_size", type=int, default=100)
  parser.add_argument("--train_date_start", default="20090101")
  parser.add_argument("--train_date_end", default="20150101")
  parser.add_argument("--test_date_start", default="20170102")
  parser.add_argument("--test_date_end", default=None)
  args = parser.parse_args()

#유틸 저장 및 경로 설정
utils.Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
utils.SAVE_DIR = "/Users/mac/Desktop/RLPortfolio" + "/" + "DirichletPortfolio"
os.makedirs(utils.SAVE_DIR + "/Metrics", exist_ok=True)
os.makedirs(utils.SAVE_DIR + "/Models", exist_ok=True)

path_list = []
for stock_code in args.stock_code:
    path = utils.Base_DIR + "/" + stock_code
    path_list.append(path)

# 학습/테스트 데이터 준비
train_data, test_data = DataManager.get_data_tensor(path_list,
                                                    train_date_start=args.train_date_start,
                                                    train_date_end=args.train_date_end,
                                                    test_date_start=args.test_date_start,
                                                    test_date_end=args.test_date_end)

# # 최소/최대 투자 가격 설정
min_trading_price = 0
# max_trading_price = 2000000
max_trading_price = 500000


# 파라미터 설정
params = {"lr":args.lr, "tau":args.tau, "K":len(args.stock_code),
          "chart_data": train_data, "discount_factor":args.discount_factor, "delta":args.delta,
          "min_trading_price": min_trading_price, "max_trading_price": max_trading_price,
          "batch_size":args.batch_size, "memory_size":args.memory_size}

# 학습/테스트 수행
train = DIRILearner(**params)
train.run(num_episode=args.num_episode, balance=args.balance)
train.save_model(critic_path=utils.SAVE_DIR + "/Models" + "/DirichletPortfolio_critic.pth",
                 actor_path=utils.SAVE_DIR + "/Models" + "/DirichletPortfolio_actor.pth",
                 score_net_path=utils.SAVE_DIR + "/Models" + "/DirichletPortfolio_score.pth")
