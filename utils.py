import DataManager

Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/content"
DATA_DIR = "/content/RLPortfolio-Dirichlet-COLAB/Data"
STOCK_LIST = None
NOW_PORT = None
NOW_PRICE = None

a = "/content/RLPortfolio-Dirichlet-COLAB/Data/HA"
b = "/content/RLPortfolio-Dirichlet-COLAB/Data/WBA"
c = "/content/RLPortfolio-Dirichlet-COLAB/Data/INCY"
d = "/content/RLPortfolio-Dirichlet-COLAB/Data/AAPL"
e = "/content/RLPortfolio-Dirichlet-COLAB/Data/COST"
f = "/content/RLPortfolio-Dirichlet-COLAB/Data/BIDU"
g = "/content/RLPortfolio-Dirichlet-COLAB/Data/TCOM"


dataset1 = [a, b, c]
dataset2 = [a, b, c, f]
dataset3 = [a, b, c, f, g]
dataset4 = [a, b, c, d]
dataset5 = [a, b, c, d, e]
dataset6 = [a, b, c, f, g, d, e]
dataset7 = [f, g, d, e]