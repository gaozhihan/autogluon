from gluonts.evaluation.backtest import make_evaluation_predictions
from ....utils.loaders import load_pickle
from ....utils.savers import save_pickle


class AbstractModel:

    def __init__(self, hyperparameters=None, model=None):
        self.set_default_parameters()
        self.params = {}
        if hyperparameters is not None:
            self.params.update(hyperparameters)
        self.model = model
        self.name = None

    def save(self, path):
        save_pickle.save(path=path, obj=self)

    @classmethod
    def load(cls, path):
        return load_pickle.load(path)

    def set_default_parameters(self):
        pass

    def create_model(self):
        pass

    def fit(self, train_ds):
        pass
        # self.model = self.model.train(train_ds)

    def predict(self, test_ds, num_samples=100):
        pass

    def hyperparameter_tune(self, train_data, test_data, scheduler_options, **kwargs):
        pass

    def score(self, y, y_true):
        pass

