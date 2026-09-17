"""Run the policy's input guards without initializing unused neural weights.

This is for input verification only, never a replacement for forward/backward
validation. require_batch still checks all feature banks and lifecycle values.
"""
from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig
from training_r0.model import R0Policy
from training_r0.observation import ObservationBuilder


class StrictInputValidator:
    def __init__(self, config=None):
        self.config=config or ModelConfig()
        if self.config.allow_incomplete_capture:
            raise ValueError('diagnostic input validator is forbidden')
        self.observation_schema_hash=ObservationBuilder(CardVocabulary.from_native(),self.config).schema_hash

    def _check_batch(self,batch):
        # Invoke the authoritative method, not a copied/reduced check list.
        return R0Policy._check_batch(self,batch)
