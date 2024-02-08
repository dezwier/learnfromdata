import unittest
import pandas as pd
import numpy as np

from lfd import Data, Encoder, set_logging


class EncoderTests(unittest.TestCase):

    @classmethod
    def setUp(self):

        self.df = pd.DataFrame([
            ["a", 5, 0, None],
            ["b", 4, 0, None],
            ["a", 0, 0, None],
            ["a", 2, 0, 0],
            ["b", -5, 1, 0],
            ["c", -5, 1, 0]
        ], columns=["var1", "var2", "var3", "var4"])
        self.data = Data(self.df.copy(), name="Test_df")

        # Logging
        set_logging(stdout=False)


    def test_encoding_allvalues(self):

        # Learn encoder
        encoder = Encoder().learn(self.data, min_occ=0)
        self.assertTrue(np.array_equal(encoder.encode_values['var1'], ["a", "b", "c"]))
        self.assertTrue(self.data.df.equals(self.df))


        # Apply encoder not inplace
        x = encoder.apply(self.data, inplace=False)
        self.assertTrue(np.array_equal(x.df.columns, ["var2", "var3", "var4", "var1__a", "var1__b", "var1__c"]))
        self.assertTrue(np.array_equal(self.data.df.columns, ["var1", "var2", "var3", "var4"]))
        self.assertTrue(self.data.df.equals(self.df))

        # Apply encoder not inplace
        encoder.apply(self.data, inplace=True)
        self.assertTrue(np.array_equal(self.data.df.columns, ["var2", "var3", "var4", "var1__a", "var1__b", "var1__c"]))


    def test_encoding_notallvalues(self):

        # Learn encoder
        encoder = Encoder().learn(self.data, min_occ=0.2)  # Value c occurs less than 20%, so not encoded
        self.assertTrue(np.array_equal(encoder.encode_values['var1'], ["a", "b", 'OTHER']))
        self.assertTrue(self.data.df.equals(self.df))


        # Apply encoder not inplace
        x = encoder.apply(self.data, inplace=False)
        self.assertTrue(np.array_equal(x.df.columns, ["var2", "var3", "var4", "var1__a", "var1__b", "var1__OTHER"]))
        self.assertTrue(np.array_equal(self.data.df.columns, ["var1", "var2", "var3", "var4"]))
        self.assertTrue(self.data.df.equals(self.df))

        # Apply encoder not inplace
        encoder.apply(self.data, inplace=True)
        self.assertTrue(np.array_equal(self.data.df.columns, ["var2", "var3", "var4", "var1__a", "var1__b", "var1__OTHER"]))


    def test_encoding_includeothers(self):

        # Learn encoder
        encoder = Encoder().learn(self.data, min_occ=0.2, include_others=True)  # Value c occurs less than 20%, so not encoded
        self.assertTrue(np.array_equal(encoder.encode_values['var1'], ["a", "b", "OTHER"]))
        self.assertTrue(self.data.df.equals(self.df))


        # Apply encoder not inplace
        x = encoder.apply(self.data, inplace=False)
        self.assertTrue(np.array_equal(x.df.columns, ["var2", "var3", "var4", "var1__a", "var1__b", "var1__OTHER"]))
        self.assertTrue(np.array_equal(self.data.df.columns, ["var1", "var2", "var3", "var4"]))
        self.assertTrue(self.data.df.equals(self.df))

        # Apply encoder not inplace
        encoder.apply(self.data, inplace=True)
        self.assertTrue(np.array_equal(self.data.df.columns, ["var2", "var3", "var4", "var1__a", "var1__b", "var1__OTHER"]))


    def test_encoding_exclude(self):

        # Learn encoder
        encoder = Encoder().learn(self.data, set_aside=['var1'])
        self.assertTrue('var1' not in encoder.encode_values)

        # Apply encoder not inplace
        x = encoder.apply(self.data, inplace=False)
        self.assertTrue(x.df.equals(self.df))
