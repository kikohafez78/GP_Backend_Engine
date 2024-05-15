# -*- coding: utf-8 -*-
"""
Not-i-within-i Feature class.
"""

from .abstract_feature import AbstractClusterFeature


class NotIWithinI(AbstractClusterFeature):

    """A class that represents the not-i-within-i Feature
    used to determine if one of the mentions is the child of the other.
    """

    def __call__(self, clusters, antecedent, mention):
        """Returns False if mention and any mention in antecedent cluster
        are in an i-within-i construction, meaning one is a child of the other.

        Args:
            clusters (Clusters):
                A Clusters object.
            antecedent (Mention):
                A Mention object that appears before the given mention.
            mention (Mention): A Mention object.

        Returns:
            True if none of the mentions in the antecedent cluster
            are in an i-within-i construction with the mention.
            False otherwise.
        """
        ant_cluster = clusters[antecedent]
        ment_cluster = clusters[mention]
        for ment in ment_cluster:
            for ant in ant_cluster:
                # A i within i construction can only appear in same sentence.
                if ment.index() == ant.index():
                    # Assuming well formed syntax trees, a mention is
                    # within the constituent of another mention
                    # if their spans overlap.
                    start1, end1 = ment.span()
                    start2, end2 = ant.span()
                    if start1 <= end2 and end1 >= start2:
                        return False
        return True
