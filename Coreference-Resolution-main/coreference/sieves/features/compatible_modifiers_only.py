# -*- coding: utf-8 -*-
"""
A class that implements the compatible modifiers feature.
"""
from .abstract_feature import AbstractMentionFeature


class CompatibleModifiersOnly(AbstractMentionFeature):

    def __init__(self, modifiers={"JJ", "JJR", "JJS", "NN", "NNP", "NNS"}):
        """Constructor of the CompatibleModifiersOnly instance.

        Args:
            modifiers:
                An iterable of strings that represent valid pos tags
                for modifiers.
        """
        self.modifiers = modifiers

    def __call__(self, antecedent, mention):
        """Determines if the modifiers of the given mention
        are all included in the modifiers of the antecedent.

        Args:
            antecedent (Mention):
                A Mention object that represents a mention that
                appears before the given mention.
            mention (Mention): A Mention object.

        Returns: Boolean
        """
        mod_ant = self._extract_modifiers(antecedent)
        mod_ment = self._extract_modifiers(mention)
        if mod_ment.issubset(mod_ant):
            return True
        return False

    def _extract_modifiers(self, mention):
        """Extracts the modifiers from the pos tagged
        leaves of a mention.

        A leaf is considered a modifier if its pos tag is
        in self.modifiers and its token is not the head noun.

        Args:
            mention (Mention): Any Mention object.

        Returns:
            A set of strings that are considered modifiers.
        """
        pos = mention.pos
        head = mention.head
        modifiers = set()
        for token, pos in pos:
            # This might lead to a modifier being excluded
            # if it has the same token as the head but
            # this works very well for most cases and is very simple.
            if pos in self.modifiers and token != head:
                modifiers.add(token)
        return modifiers
