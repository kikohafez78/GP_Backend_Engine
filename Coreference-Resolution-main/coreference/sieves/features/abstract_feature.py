# -*- coding: utf-8 -*-
"""
Abstract Class for Features:
    - a MentionFeature only considers two mention objects.
    - a ClusterFeature needs additional informations from a Clusters object.
"""
from abc import ABC, abstractmethod


class AbstractClusterFeature(ABC):

    @abstractmethod
    def __call__(self, clusters, antecedent, mention):
        """This method determines if the concrete feature applies
        to two clusters of mentions. It should return a Boolean value and
        take a cluster and two mentions as input.
        """
        return False


class AbstractMentionFeature(ABC):

    @abstractmethod
    def __call__(self, antecedent, mention):
        """This method also determines if a feature applies but
        it should only take two mentions as input. It should return
        a Boolean Value"""
        return False
