import pandas as pd

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(url, header=0, index_col=0)

from gluonts.dataset.common import ListDataset

training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq="5min",
)

from gluonts.model.deepar_onestep import OneStepDeepAREstimator
from gluonts.model.forecast import DistributionForecast
from gluonts.trainer import Trainer

estimator = OneStepDeepAREstimator(
    freq="5min", context_length=36, trainer=Trainer(epochs=3)
)
predictor = estimator.train(training_data=training_data)

test_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-15 00:00:00"]}],
    freq="5min",
)

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    assert isinstance(forecast, DistributionForecast)
    print(forecast.start_date)
    print(forecast.distribution)
