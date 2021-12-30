
from __future__ import annotations
import logging
from typing import List, Type, Dict, Text, Any, Optional, Tuple
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
import numpy as np

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
import rasa.utils.io as io_utils
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.storage import ModelStorage
from rasa.core.policies.policy import Policy, PolicyPrediction, SupportedData
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message

from .sklearn_utils import SklearnUtils
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True 
)

class SVCPolicy(Policy):
    """Interface for any component which will run in a graph."""

    def __init__(self, 
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        clf = None,
        encoder = None,
        featurizer: Optional[TrackerFeaturizer] = None) -> None:

        super().__init__(config, model_storage, resource, execution_context, featurizer=featurizer)

        self.sk_utils = SklearnUtils()
        self.clf      = clf
        self.encoder  = encoder

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []
    
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config.

        Default config and user config are merged by the `GraphNode` before the
        config is passed to the `create` and `load` method of the component.

        Returns:
            The default config of the component.
        """
        return {
            'enable_feature_string_compression': True,
            'use_nlu_confidence_as_score': False,
            'priority': 3,
            'max_history': None,
            # C parameter of the svm - cross validation will select the best value
            "C": [10, 20, 100],
            # gamma parameter of the svm
            "gamma": [0.1],
            # the kernels to use for the svm training - cross validation will
            # decide which one of them performs best
            "kernels": ["linear", "rbf"],
            # We try to find a good number of cross folds to use during
            # intent training, this specifies the max number of folds
            "max_cross_validation_folds": 5,
            # Scoring function used for evaluating the hyper parameters
            # This can be a name or a function (cfr GridSearchCV doc for more info)
            "scoring_function": "f1_weighted",
            "num_threads": 3,
        }


    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SVCPolicy:
        """Creates a new `GraphComponent`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.

        Returns: An instantiated `GraphComponent`.
        """
        return cls(config, model_storage, resource, execution_context)


    def train(
        self, 
        training_trackers: List[TrackerWithCachedStates], 
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any) -> Resource:

        x_train, y_train = self.sk_utils.transforms(training_trackers)

        self.encoder = self.sk_utils.get_encoder()

        self.clf = self._create_classifier(
            self.config['num_threads'],
            y_train
        ).fit(x_train, y_train)
        
        self.persist()
        return self._resource

    def _create_classifier(
        self, num_threads: int, y: np.ndarray
    ):

        C = self.config["C"]
        kernels = self.config["kernels"]
        gamma = self.config["gamma"]
        # dirty str fix because sklearn is expecting
        # str not instance of basestr...
        tuned_parameters = [
            {"C": C, "gamma": gamma, "kernel": [str(k) for k in kernels]}
        ]

        return GridSearchCV(
            SVC(C=1, probability=True, class_weight="balanced"),
            param_grid=tuned_parameters,
            n_jobs=num_threads,
            scoring=self.config["scoring_function"],
            verbose=1
        )

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as model_dir:
            file_name = self.__class__.__name__
            classifier_file_name = model_dir / f"{file_name}_classifier.pkl"
            encoder_file_name = model_dir / f"{file_name}_encoder.pkl"

            if self.clf:
                io_utils.json_pickle(classifier_file_name, self.clf.best_estimator_)
                io_utils.json_pickle(encoder_file_name, self.encoder)

    def _model_report(self, x_test, y_test):
        yp_class = self.clf.predict(x_test)
        print(metrics.classification_report(y_test, yp_class))

    def predict_action_probabilities(
        self, 
        tracker: DialogueStateTracker, 
        domain: Domain, 
        rule_only_data: Optional[Dict[Text, Any]] = None, 
        **kwargs: Any) -> PolicyPrediction:


        text = tracker.current_state(domain)['latest_message']['text']
        X = self.encoder.transform(
            [self.sk_utils._clean_text(text)]
        ).toarray()
        y_pred = self.clf.predict(X)[0]
        probabilities = self.clf.predict_proba(X)[0]

        return PolicyPrediction(list(probabilities), y_pred)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> SVCPolicy:
        """Creates a component using a persisted version of itself.

        If not overridden this method merely calls `create`.

        Args:
            config: The config for this graph component. This is the default config of
                the component merged with config specified by the user.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            kwargs: Output values from previous nodes might be passed in as `kwargs`.

        Returns:
            An instantiated, loaded `GraphComponent`.
        """
        try:
            with model_storage.read_from(resource) as model_dir:
                file_name = cls.__name__
                classifier_file = model_dir / f"{file_name}_classifier.pkl"

                if classifier_file.exists():
                    classifier = io_utils.json_unpickle(classifier_file)

                    encoder_file = model_dir / f"{file_name}_encoder.pkl"
                    encoder = io_utils.json_unpickle(encoder_file)

                    return cls(config, model_storage, resource, execution_context, classifier, encoder)
        except ValueError:
            logger.debug(
                f"Failed to load '{cls.__name__}' from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
        return cls.create(config, model_storage, resource, execution_context)

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component can work with.

        Returns: A list of supported languages, or `None` to signify all are supported.
        """
        return None

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component cannot work with.

        Returns: A list of not supported languages, or
            `None` to signify all are supported.
        """
        return None

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return []