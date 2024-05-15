# -*- coding: utf-8 -*-
"""
A class that represents the word inclusion feature
"""
from nltk.corpus import stopwords

from .abstract_feature import AbstractClusterFeature


class WordInclusion(AbstractClusterFeature):

    """A class that implements the word inclusion feature.
    Because it is assumed that a mention contains no more novel information
    than its antecedent, the set of non-stopwords of the mention cluster
    should be a subset of the set of non-stopwords of the antecedent cluster.
    NLTK's stopword corpus is used.
    """

    def __init__(self, lang="english"):
        """Constructor of WordInclusion instance.

        Args:
            lang (str): The language of which stopwords should be used.

        Raises:
            NotImplementedError if the given language is not supported.
        """
        if lang not in stopwords.fileids():
            raise NotImplementedError(f"Unknown language: {lang}")
        self.stopwords = stopwords.words(lang)

    def __call__(self, clusters, antecedent, mention):
        """Checks if all non-stopwords of the given mention cluster
        are included in the set of non-stopwords of the given antecedent
        cluster.

        Args:
            clusters (Clusters):
                A Clusters object.
            antecedent (Mention):
                A Mention object that represents a mention that
                appears before the given mention.
            mention (Mention):
                A Mention object.

        Returns:
            Boolean
        """
        non_stops_ment = self._get_non_stopwords(clusters[mention])
        non_stops_ant = self._get_non_stopwords(clusters[antecedent])
        if non_stops_ment.issubset(non_stops_ant):
            return True
        return False

    def _get_non_stopwords(self, cluster):
        """Extract words that are not stopwords from a cluster.

        Args:
            cluster (set): A set of Mention objects.

        Returns:
            Set of strings that are not considered stopwords.
        """
        non_stops = set()
        for mention in cluster:
            for word in mention.words:
                if word not in self.stopwords:
                    non_stops.add(word)
        return non_stops
