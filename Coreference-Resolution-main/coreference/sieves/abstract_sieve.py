# -*- coding: utf-8 -*-
"""
An abstract class for sieves.
"""

from abc import ABC, abstractmethod

from .features.abstract_feature import AbstractMentionFeature, AbstractClusterFeature


class AbstractSieve(ABC):

    def apply_features(self, features, clusters, antecedent, mention):
        """For an iterable of Feature objects checks for each features
        if it applies depending on whether it is Cluster of Mention feature.

        Args:
            features: An iterable of Feature objects.
            clusters: A Clusters object.
            antecedent:
                A Mention object that represents a mention appearing
                before the given mention.
            mention: Any Mention object.

        Returns:
            A list of boolean values.

        Raises:
            TypeError if one of the objects in features does
            not inherit from AbstractMentionFeature or AbstractClusterFeature.
        """
        applies = []
        for feature in features:
            if isinstance(feature, AbstractMentionFeature):
                applies.append(feature(antecedent, mention))
            elif isinstance(feature, AbstractClusterFeature):
                applies.append(feature(clusters, antecedent, mention))
            else:
                raise TypeError("Features must inherit "
                                "from AbstractMentionFeature or "
                                "AbstractClusterFeature")
        return applies

    @abstractmethod
    def resolve(self, clusters):
        """Method that resolves mentions in a Cluster object.
        Mentions are resolved if features apply.

        Args:
            clusters (Clusters): Any Cluster object.
        """
        return
