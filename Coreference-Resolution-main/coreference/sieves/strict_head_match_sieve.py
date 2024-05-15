# -*- coding: utf-8 -*-
"""
Strict Head Matching Sieve
"""

from .abstract_sieve import AbstractSieve
from .features.cluster_head_match import ClusterHeadMatch
from .features.compatible_modifiers_only import CompatibleModifiersOnly
from .features.not_i_within_i import NotIWithinI
from .features.word_inclusion import WordInclusion


class StrictHeadMatchSieve(AbstractSieve):

    """A class that implements the strict head match sieve.
    This class has the features ClusterHeadMatch, WordInclusion,
    NotIWithinI, CompatibleModifiersOnly.
    All of these features have to apply for two mentions to be merged.
    """

    def __init__(self,
                 lang="english",
                 modifiers={"JJ", "JJR", "JJS", "NN", "NNP", "NNS"}):
        """Constructor of the StrictHeadMatchSieve instance.

        Args:
            lang (str): The language that should be used for stopwords.
            modifiers: Iterable of valid tags for modifiers.
        """
        self.features = [ClusterHeadMatch(),
                         WordInclusion(lang),
                         NotIWithinI(),
                         CompatibleModifiersOnly(modifiers)]

    def resolve(self, clusters):
        unresolved = clusters.unresolved()
        for mention in unresolved:
            # Ignore pronominal and indefinite mentions.
            if not mention.pronominal and not mention.indefinite:
                for antecedent in clusters.antecedents(mention):
                    if not antecedent.pronominal and not antecedent.indefinite:
                        apply_all = self.apply_features(self.features,
                                                        clusters,
                                                        antecedent,
                                                        mention)
                        # Sieve is conjunctive, all features have to be True
                        # for two mentions to be merged.
                        if all(apply_all):
                            clusters.merge(antecedent, mention)
                            break
