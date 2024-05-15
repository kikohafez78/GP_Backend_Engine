# -*- coding: utf-8 -*-
"""
A class that represents a variant of the StrictHeadMatchSieve.
It is missing the compatible word inclusion feature.
"""
from .strict_head_match_sieve import StrictHeadMatchSieve
from .features.cluster_head_match import ClusterHeadMatch
from .features.compatible_modifiers_only import CompatibleModifiersOnly
from .features.not_i_within_i import NotIWithinI


class StrictHeadRelaxInclusion(StrictHeadMatchSieve):

    """A class that implements a variant of the strict head sieve.
    It has the features ClusterHeadMatch, CompatibleModifiers
    and NotIWithinI."""

    def __init__(self, modifiers={"JJ", "JJR", "JJS", "NN", "NNP", "NNS"}):
        """Constructor of the StrictHeadRelaxInclusion instance.

        Args:
            modifers:
                Iterable of string that represent valid tags for modifiers.
        """
        self.features = [ClusterHeadMatch(),
                         NotIWithinI(),
                         CompatibleModifiersOnly(modifiers)]
