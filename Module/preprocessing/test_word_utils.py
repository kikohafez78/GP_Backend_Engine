import pytest

from .word_utils import get_inflected_noun, levenshtein_distance, SimpleSpellCorrector

def test_inflect_noun_simple():
    cases = ['cars','houses','books','birds','pencils']
    expected = ['car', 'house', 'book','bird','pencil']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_noun_1rule():
    cases = ['kisses','wishes','matches','foxes','quizzes']
    expected = ['kiss', 'wish', 'match','fox','quiz']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_noun_2rule():
    cases = ['boys','holidays','keys','guys']
    expected = ['boy', 'holiday', 'key','guy']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_noun_3rule():
    cases = ['parties','ladies','stories','nannies', 'cities']
    expected = ['party','lady','story','nanny', 'city']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_noun_4rule():
    cases = ['lives','leaves','thieves','wives']
    expected = ['life','leaf','thief','wife']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_noun_5rule():
    cases = ['tomatoes','potatoes','echoes','heroes']
    expected = ['tomato','potato','echo','hero']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_noun_irergular1():
    cases = ['men','women','children','feet','teeth','geese','mice']
    expected = ['man','woman','child','foot','tooth','goose','mouse']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_noun_irergular2():
    cases = ['fish','sheep','deer','moose','aircraft']
    expected = ['fish','sheep','deer','moose','aircraft']
    obtained = [get_inflected_noun(word) for word in cases]
    assert expected == obtained

def test_inflect_empty():
    cases = ''
    expected = ''
    assert get_inflected_noun(cases) == expected

def test_inflect_num():
    cases = 3
    expected = '3'
    assert get_inflected_noun(cases) == expected

# Levensthein distance
def test_levenshtein():
    cases = [('cat','cats'),('zip','zipper'),('a','a'), ('words with space', 'word with space')]
    expected = [1,3,0,1]
    for idx in range(len(cases)):
        assert levenshtein_distance(cases[idx][0], cases[idx][1]) == expected[idx]

def test_levenshtein_empty():
    case = ('','')
    expected = 0
    assert levenshtein_distance(case[0],case[1])==expected

def test_levenshtein_numbers():
    case = (1,'')
    expected = 1
    assert levenshtein_distance(case[0],case[1])==expected
