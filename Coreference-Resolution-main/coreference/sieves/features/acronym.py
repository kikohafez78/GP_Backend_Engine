# -*- coding: utf-8 -*-
"""
A feature that determines if two mentions are acronyms of each other.
"""
from .abstract_feature import AbstractMentionFeature


class Acronyms(AbstractMentionFeature):

    def __init__(self, tags=("NNP",)):
        """Constructor of an Acronyms Instance.

        Args:
            tags:
                Iterable that contains strings representing
                tags that can be considered proper nouns.
        """
        self.tags = tags

    def __call__(self, mention1, mention2):
        """Returns True if one of the mentions is an acronym of the other.

        A mention is an acronym of the other if both are tagged as a proper
        nouns and the uppercase letters in one are the text of the other.

        Args:
            mention1 (Mention)
            mention2 (Mention)

        Returns:
            True if they are acronoyms. False otherwise.
        """
        words1 = mention1.words
        words2 = mention2.words
        if self._is_proper(mention1) and self._is_proper(mention2):
            upper1 = self._upper_letters(words1)
            upper2 = self._upper_letters(words2)
            if words1 == upper2 or words2 == upper1:
                return True
        return False

    def _is_proper(self, mention):
        """Determines if all pos tags in mention are proper noun tags."""
        return all(pos in self.tags for _, pos in mention.pos)

    def _upper_letters(self, words):
        """Extracts all upper letters, joins them in a string
        and returns this string in a list."""
        upper = (letter for w in words for letter in w if letter.isupper())
        return ["".join(upper)]
