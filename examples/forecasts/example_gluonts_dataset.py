from gluonts.dataset.repository.datasets import get_dataset
from autogluon.forecasting import Forecasting as task

dataset = get_dataset("m4_hourly", regenerate=True)
train_data = dataset.train
test_data = dataset.test

prediction_length = dataset.metadata.prediction_length
freq = dataset.metadata.freq
index_column = None
target_column = None
time_column = None

predictor = task.fit(train_data=train_data,
                     prediction_length=prediction_length,
                     index_column=index_column,
                     target_column=target_column,
                     time_column=time_column,
                     freq=freq,
                     )

print(predictor.leaderboard())
print(predictor.evaluate(test_data))