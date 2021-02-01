from autogluon.core.task.base.base_task import BaseTask, schedulers, compile_scheduler_options
from .dataset import TimeSeriesDataset
from autogluon.core.dataset import TabularDataset
from ...utils.dataset_utils import rebuild_tabular, train_test_split_dataframe, train_test_split_gluonts
from ...learner import DefaultLearner as Learner
from .predictor import ForecastingPredictor
from ...trainer import AutoTrainer
import pandas as pd
from gluonts.dataset.common import Dataset, ListDataset, FileDataset
from autogluon.core.utils.utils import setup_outputdir
__all__ = ['Forecasting']


class Forecasting(BaseTask):

    Dataset = TabularDataset
    Predictor = ForecastingPredictor

    @staticmethod
    def load(output_directory, verbosity=2):
        return ForecastingPredictor.load(output_directory=output_directory, verbosity=verbosity)

    @staticmethod
    def fit(train_data,
            prediction_length,
            index_column="index",
            target_column=None,
            time_column="date",
            val_data=None,
            time_limits=None,
            output_directory=None,
            eval_metric=None,
            hyperparameter_tune=False,
            hyperparameters=None,
            num_trials=None,
            search_strategy='random',
            verbosity=2,
            **kwargs):

        # TODO: allow user to input more scheduler options and use compile_scheduler_options()
        output_directory = setup_outputdir(output_directory)
        freq = kwargs.get("freq", None)
        if isinstance(train_data, pd.DataFrame):
            train_data = rebuild_tabular(train_data,
                                         index_column=index_column,
                                         target_column=target_column,
                                         time_column=time_column)
            if val_data is None:
                train_data, val_data = train_test_split_dataframe(train_data, prediction_length)
            else:
                val_data = rebuild_tabular(val_data,
                                           index_column=index_column,
                                           target_column=target_column,
                                           time_column=time_column)
            train_data = TimeSeriesDataset(train_data, index_column=index_column)
            freq = train_data.get_freq()
            if val_data is not None:
                val_data = TimeSeriesDataset(val_data, index_column=index_column)
        elif isinstance(train_data, FileDataset) or isinstance(train_data, ListDataset):
            if val_data is None:
                train_data, val_data = train_test_split_gluonts(train_data, prediction_length, freq)
        else:
            raise TypeError("Does not support dataset type:", type(train_data))
        trainer_type = kwargs.get('trainer_type', AutoTrainer)
        random_seed = kwargs.get('random_seed', 0)
        quantiles = kwargs.get("quantiles", ["0.5"])
        scheduler_options = {"searcher": search_strategy,
                             "resource": {"num_cpus": 1, "num_gpus": 0},
                             "num_trials": num_trials,
                             "reward_attr": "validation_performance",
                             "time_attr": "epoch",
                             "time_out": time_limits}

        scheduler_cls = schedulers[search_strategy.lower()]

        scheduler_options = (scheduler_cls, scheduler_options)
        learner = Learner(path_context=output_directory,
                          eval_metric=eval_metric,
                          trainer_type=trainer_type,
                          random_seed=random_seed,)

        learner.fit(train_data=train_data,
                    freq=freq,
                    prediction_length=prediction_length,
                    val_data=val_data,
                    scheduler_options=scheduler_options,
                    hyperparameters=hyperparameters,
                    hyperparameter_tune=hyperparameter_tune,
                    quantiles=quantiles)

        # TODO: refit full
        predictor = ForecastingPredictor(learner,
                                         index_column=index_column,
                                         target_column=target_column,
                                         time_column=time_column)
        predictor.save(output_directory)
        return predictor





