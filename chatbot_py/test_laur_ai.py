import unittest
from pandas import read_csv
from laur_ai import LaurAI

'''
TEST Laur.ai class

TODO: move to separate directory
'''

class TestLauAI:
    '''
    test 
    '''
    def setup_test(self):
        data = read_csv("data/master_data.csv")
        self.laurbot = LaurAI(data)
        self.laurbot.create_lemma()
        self.laurbot.create_bag_of_words()

    def test_create_lemma(self):
        self.laurbot.create_lemma()
        if not self.laurbot.finalText().empty():
            assert True
        else:
            assert False


if __name__ == "__main__":
     unittest.main(argv=[''], verbosity=2, exit=False)