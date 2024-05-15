# -*- coding: utf-8 -*-
"""
A partial implementation of the precise construct sieve.
"""

from .abstract_sieve import AbstractSieve
from .features.acronym import Acronyms
from .features.appositive import Appositive
from .features.predicate_nominative import PredicateNominative


class PreciseConstructsSieve(AbstractSieve):

    """This is a partial implementation of the precise construct sieve.
    It has the features PredicateNominative, Acronyms and Appositive.

    Raghunathan et. al. describes two more features for this sieve:
        - role appositive:
            Ignored here because of missing information and because
            Appositive feature already covers most cases.
        - relative pronoun:
            Ignored here because relative pronouns are almost never
            tagged as coreferential in Ontonotes corpus.
    """

    def __init__(self, predicate={"be"}):
        """Constructor of PreciseConstructsSieve instance.

        Args:
            predicate:
                Iterable of strings that represent valid predicates
                for PredicateNominative feature.
        """
        self.features = [PredicateNominative(predicate),
                         Acronyms(),
                         Appositive()]

    def resolve(self, clusters):
        unresolved = clusters.unresolved()
        for mention in unresolved:
            if not mention.indefinite:
                for antecedent in clusters.antecedents(mention):
                    if not antecedent.indefinite:
                        apply_all = self.apply_features(self.features,
                                                        clusters,
                                                        antecedent,
                                                        mention)
                        # Sieve is disjunctive: if any feature
                        # applies, two mentions can be merged.
                        if any(apply_all):
                            clusters.merge(antecedent, mention)
                            break
