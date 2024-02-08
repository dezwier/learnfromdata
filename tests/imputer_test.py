import unittest
import pandas as pd
import numpy as np
from lfd import Data, Imputer, set_logging


class ImputerTests(unittest.TestCase):

    @classmethod
    def setUp(self):

        self.df = pd.DataFrame([
            ["a", 5, 9, -9],
            ["d", None, 0, 5],
            ["c", 0, 0, 0],
            [None, -5, None, 0]
        ], columns=["var1", "var2", "var3", "var4"])
        self.data = Data(self.df.copy(), name="Test_df")

        # Logging
        set_logging(stdout=False)


    def test_imputer_mean(self):

        # Fit splitter
        self.assertEqual(self.data.df.isna().sum().sum(), 3)
        imputer = Imputer()
        imputer.learn(self.data, default_cat='missing', default_cont='mean')

        # Test attributes and if data is untouched
        self.assertEqual(imputer.impute_values["var1"], "missing")
        self.assertEqual(imputer.impute_values["var2"], 0)
        self.assertEqual(imputer.impute_values["var3"], 3)
        self.assertTrue(self.data.df.equals(self.df))

        # Transform
        x = imputer.apply(self.data)

        # Test X and if if data is untouched
        self.assertEqual(x.df.iloc[1, 1], 0)
        self.assertEqual(x.df.iloc[3, 0], "missing")
        self.assertTrue(self.data.df.equals(self.df))

        # Assert expected missing values
        self.assertEqual(self.data.df.isna().sum().sum(), 3)
        self.assertEqual(x.df.isna().sum().sum(), 0)


    def test_imputer_median(self):

        # Fit splitter
        self.assertEqual(self.data.df.isna().sum().sum(), 3)
        imputer = Imputer()
        imputer.learn(self.data, default_cat='<MISSING>', default_cont='median')

        # Test attributes and if data is untouched
        self.assertEqual(imputer.impute_values["var1"], "<MISSING>")
        self.assertEqual(imputer.impute_values["var2"], 0)
        self.assertEqual(imputer.impute_values["var3"], 0)
        self.assertTrue(self.data.df.equals(self.df))

        # Transform
        x = imputer.apply(self.data)

        # Test X and if if data is untouched
        self.assertEqual(x.df.iloc[1, 1], 0)
        self.assertEqual(x.df.iloc[3, 0], "<MISSING>")
        self.assertTrue(self.data.df.equals(self.df))

        # Assert expected missing values
        self.assertEqual(self.data.df.isna().sum().sum(), 3)
        self.assertEqual(x.df.isna().sum().sum(), 0)



    def test_imputer_inplace(self):

        # Fit splitter
        imputer = Imputer()
        imputer.learn(self.data, default_cat='missing', default_cont='mean')

        # Test if data is untouched
        imputer.apply(self.data, inplace=False)
        self.assertTrue(self.data.df.equals(self.df))

        # Test data inplace
        imputer.apply(self.data, inplace=True)
        self.assertEqual(self.data.df.iloc[1, 1], 0)
        self.assertEqual(self.data.df.iloc[3, 0], "missing")

        # Assert expected missing values
        self.assertEqual(self.data.df.isna().sum().sum(), 0)

