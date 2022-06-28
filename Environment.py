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

