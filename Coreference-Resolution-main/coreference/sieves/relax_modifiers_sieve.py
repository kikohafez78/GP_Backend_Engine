# -*- coding: utf-8 -*-
"""
A class that represents a variant of the StrictHeadMatchSieve.
It is missing the compatible modifier feature.
"""
from .strict_head_match_sieve import StrictHeadMatchSieve
from .features.cluster_head_match import ClusterHeadMatch
from .features.not_i_within_i import NotIWithinI
from .features.word_inclusion import WordInclusion


class StrictHeadRelaxModifiers(StrictHeadMatchSieve):

    """A class that implements a variant of the strict head sieve.
    It has the features ClusterHeadMatch, WordInclusion and NotIWithinI."""

    def __init__(self, lang="english"):
        """Constructor of the StrictHeadRelaxModifiers instance.

        Args:
            lang (str): The language that should be used for stopwords.
        """
        self.features = [ClusterHeadMatch(),
                         WordInclusion(lang),
                         NotIWithinI()]
