# -*- coding: utf-8 -*-
"""
A class for the Appositive feature.
"""
from nltk.tree import Tree

from .abstract_feature import AbstractMentionFeature


class Appositive(AbstractMentionFeature):

    """A feature class that represents an appositive construction
    like 'the president, Joe Biden'.
    """

    def __init__(self, allowed_labels=(",",), allowed_len=3):
        """Constructor of an Appositive instance.

        Args:
            allowed_labels:
                Iterable that contains labels that are
                additionally allowed in an appositive construction.
            allowed_len (int):
                The maximum lenght an appositive construction is
                allowed to have.

        Raises:
            ValueError if allowed_len is less than 2.
        """
        self.allowed_labels = allowed_labels
        if allowed_len < 2:
            raise ValueError("Appostive Construction "
                             "must allow at least 2 children.")
        self.allowed_len = allowed_len

    def __call__(self, antecedent, mention):
        """Returns True if antecedent and mention
        are in an appositive construction.

        An appositive construction is assumed if the
        two mentions have the same parent that has no
        more children than the allowed length and each of these
        children are either the mention/antecedent or have an
        allowed label.

        Returns:
            True if all of the above apply. False otherwise.
        """
        ment_tree = mention.tree
        ant_tree = antecedent.tree
        # Possible only in same sentence.
        if antecedent.index() == mention.index():
            parent_ant = ant_tree.parent()
            parent_ment = ment_tree.parent()
            # We might be at the root when calling parent().
            if not (parent_ant is None or parent_ment is None):
                # This means they have the same parent.
                # By using "is" it is ensured parents are the same object,
                # and don't just appear the same.
                if parent_ant is parent_ment:
                    # We allow for additional children
                    # according to the instance attributes.
                    if len(parent_ant) <= self.allowed_len:
                        appositive = True
                        for child in parent_ant:
                            is_ant = child is ant_tree
                            is_ment = child is ment_tree
                            is_allowed = False
                            if isinstance(child, Tree):
                                is_allowed = child.label() in self.allowed_labels
                            if not (is_ant or is_ment or is_allowed):
                                appositive = False
                        if appositive:
                            return True
        return False
