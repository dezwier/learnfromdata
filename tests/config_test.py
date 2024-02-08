import unittest
import os

from lfd import generate_doc, get_params


class ConfigTests(unittest.TestCase):

    def test_generate_doc(self):

        # Learn encoder
        test_file = 'doc_test.html'
        generate_doc(test_file)
        self.assertTrue(test_file in os.listdir())
        os.remove(test_file)

    def test_get_params(self):
        params = get_params()
        self.assertTrue(type(params) is dict)
