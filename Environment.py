class environment:
    PRICE_COLUMN = -1  #종가의 인덱스

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = - 1

    def reset(self):
        self.observation = None
        self.idx = - 1

    def observe(self):
        if len(self.chart_data)-1 >= self.idx:
            self.idx += 1
            self.observation = self.chart_data[self.idx]
            self.observation_train = self.observation[:environment.PRICE_COLUMN] #Price Column 제외하고 train
            return self.observation_train.transpose()
        else:
            return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[environment.PRICE_COLUMN]
        return None

if __name__ == "__main__":
    import DataManager
    path1 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/010140" #삼성중공업
    path2 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/013570" #디와이
    path3 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/010690" #화신
    path4 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/000910" #유니온
    path5 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/010060" #OCI
    path6 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/034220" #LG디스플레이
    path7 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/009540" #한국조선해양

    path_list = [path1, path2, path3, path4, path5, path6, path7]
    train_data, test_data =DataManager.get_data_tensor(path_list,
                                            train_date_start="20090101",
                                            train_date_end="20180101",
                                            test_date_start="20180102",
                                            test_date_end=None)

    env = environment(chart_data=train_data)
    env.reset()
    state = env.observe()
    print(state.shape)
    print(env.get_price())
    print(state)

