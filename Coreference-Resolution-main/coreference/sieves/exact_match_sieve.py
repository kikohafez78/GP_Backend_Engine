# -*- coding: utf-8 -*-
"""
A class that represents the exact sieve.
It resolves cluster by linking mentions if they are the same string.
"""
from .abstract_sieve import AbstractSieve


class ExactMatchSieve(AbstractSieve):

    def resolve(self, clusters):
        """Resolves mentions if they are the exact same sequence
        by merging their clusters.

        Args:
            clusters (Clusters): A Clusters object to be resolved.

        Returns: None
        """
        unresolved = clusters.unresolved()
        for mention in unresolved:
            # Sieve only operates on nominal mentions.
            if not mention.pronominal:
                # Iterates first over antecedents from same,
                # then from previous sentence.
                for mention_ant in clusters.antecedents(mention):
                    if self.is_exact_match(mention_ant, mention):
                        clusters.merge(mention_ant, mention)
                        # mention is now assigned to a cluster, so we
                        # don't need to look any further.
                        break

    def is_exact_match(self, mention1, mention2):
        """Checks if the token of two mentions equal eachother.

        Args:
            mention1 (Mention): Any Mention object.
            mention2 (Mention): Any Mention object.

        Returns:
            True if their sequences are equal, False otherwise.
        """
        if mention1.words == mention2.words:
            return True
        return False
