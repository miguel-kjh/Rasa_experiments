
from __future__ import annotations
import logging
from typing import List, Type, Dict, Text, Any, Optional, Tuple
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
import random

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.storage import ModelStorage
from rasa.core.policies.policy import Policy, PolicyPrediction, SupportedData
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True 
)

class RandomPolicy(Policy):
    """Interface for any component which will run in a graph."""

    def __init__(self, 
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None) -> None:

        super().__init__(config, model_storage, resource, execution_context, featurizer=featurizer)

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
            'max_history': None
        }


    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RandomPolicy:
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

        print(training_trackers[0].as_dialogue().as_dict())
               
        return self._resource

    def process(self, messages: List[Message]) -> List[Message]:
        # This is the method which Rasa Open Source will call during inference.
        print("msg", messages)
        return messages

    def predict_action_probabilities(
        self, 
        tracker: DialogueStateTracker, 
        domain: Domain, 
        rule_only_data: Optional[Dict[Text, Any]] = None, 
        **kwargs: Any) -> PolicyPrediction:


        print(tracker.past_states(domain))

        action     = random.choice(domain.as_dict()['actions'])
        index_act  = domain.as_dict()['actions'].index(action)
        len_action = len(domain.as_dict()['actions'])

        prediction    = [0.0]*len_action
        prediction[index_act] = 1.0
        
        return PolicyPrediction(prediction, action)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> RandomPolicy:
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
