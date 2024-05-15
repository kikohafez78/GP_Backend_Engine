# -*- coding: utf-8 -*-
"""
A class that represents the Cluster Head Matching feature.
"""

from .abstract_feature import AbstractClusterFeature


class ClusterHeadMatch(AbstractClusterFeature):

    """A Feature that represents the cluster head match."""

    def __call__(self, cluster, antecedent, mention):
        """Returns True if the head noun of the mention is
        an element in the set of head nouns of the antecedent cluster.

        Args:
            cluster (Clusters):
                A Clusters object.
            antecedent (Mention):
                A Mention object that appears before the given mention.
            mention (Mention):
                A Mention object.
        """
        head_ment = mention.head
        # Get cluster of the antecedent.
        cluster_ant = cluster[antecedent]
        # Generate the set of all head nouns in the antecedent cluster.
        heads_ant = {mention.head for mention in cluster_ant}
        if head_ment in heads_ant:
            return True
        return False
