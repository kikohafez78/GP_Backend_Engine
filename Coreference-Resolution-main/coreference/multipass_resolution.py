# -*- coding: utf-8 -*-
"""
Coreference resolution
"""
import csv

from .sieves.abstract_sieve import AbstractSieve
from .clusters import Clusters


class MultiPassResolution:

    def __init__(self, doc, sieves):
        """Constructor of MultiPassResolution instance.

        Args:
            doc (Document):
                A Document object from which coreferential mention should
                be extracted.
            sieve:
                An iterable of sieves that inherit from AbstractSieve.

        Raises:
            TypeError if a sieve does not inherit from AbstractSieve
        """
        for sieve in sieves:
            if not isinstance(sieve, AbstractSieve):
                raise TypeError("Sieves must inherit from AbstractSieve")
        self.sieves = sieves
        self.doc = doc
        self.clusters = Clusters(self.doc)

    def resolve(self):
        """Applies all sieves to clusters."""
        for sieve in self.sieves:
            sieve.resolve(self.clusters)

    def extracted_pairs(self):
        """Extracts pairs of coreferential mentions
        in current clusters.

        Returns:
            A set of pairs where each element in the pair
            is a 3-tuple of sentence index, start and end.
        """
        pairs = set()
        for mention in self.clusters:
            # Sorting ensures that we get the same pairs in gold and
            # extracted.
            cluster = sorted(self.clusters[mention], key=lambda m: m.id)
            cluster_len = len(cluster)
            for i in range(cluster_len):
                for j in range(i+1, cluster_len):
                    pair = cluster[i].id, cluster[j].id
                    pairs.add(pair)
        return pairs

    def gold_pairs(self):
        """Extracts pairs of coreferential mentions
        specified in the instances document.

        Returns:
            A set of pairs where each element in the pair
            is a 3-tuple of sentence index, start and end.
        """
        pairs = set()
        coref = self.doc.coreference_chains()
        for coref_id in coref:
            # Sorting ensures that we get the same pairs in gold and
            # extracted.
            cluster = sorted(coref[coref_id])
            cluster_len = len(cluster)
            for i in range(cluster_len):
                for j in range(i+1, cluster_len):
                    pair = cluster[i], cluster[j]
                    pairs.add(pair)
        return pairs

    def evaluate(self):
        """Computes precision, recall and f1 score based
        on pairwise comparisons of mentions in current clusters
        and mentions specified in instance document.

        Returns:
            3-tuple of floats: (precision, recall, f1)
        """
        extracted = self.extracted_pairs()
        golds = self.gold_pairs()
        true = golds.intersection(extracted)
        n_true = len(true)
        n_golds = len(golds)
        n_extracted = len(extracted)
        precision = 0
        recall = 0
        f1 = 0
        if n_extracted:
            precision = n_true / n_extracted
        if n_golds:
            recall = n_true / n_golds
        if recall or precision:
            f1 = (2*precision*recall) / (precision+recall)
        return precision, recall, f1

    def to_csv(self, path):
        """Writes found clusters to csv file.

        First line contains path to document. Clusters are seperated
        by a line with '-'. Each line represent a mention with the first
        column specifing its id and the second its words.
        """
        with open(path, "w", encoding="utf-8", newline="") as csv_f:
            csv_writer = csv.writer(csv_f, delimiter=";")
            csv_writer.writerow([self.doc.path])
            for repr_ment in self.clusters:
                cluster = self.clusters[repr_ment]
                if len(cluster) > 1:
                    for mention in cluster:
                        line = [mention.id, " ".join(mention.words)]
                        csv_writer.writerow(line)
                    csv_writer.writerow(["-", "-"])
