import unittest
import pandas as pd
import numpy as np

from lfd import Data, UniSelector, set_logging


class UnivariateTests(unittest.TestCase):

    @classmethod
    def setUp(self):

        self.df = pd.DataFrame([
            ["a", 5, 0, None],
            ["b", 4, 0, None],
            ["c", 0, 0, None],
            ["d", 2, 0, 0],
            ["e", -5, 1, 0]
        ], columns=["var1", "var2", "var3", "var4"])
        self.data = Data(self.df.copy(), name="Test_df")

        # Logging
        set_logging(stdout=False)


    def test_univariate_all(self):

        # Learn uniselector
        selector = UniSelector()
        selector.learn(self.data, min_occ=0.0, max_occ=1)
        x = selector.apply(self.data, inplace=False)

        # Test attributes and if data is untouched
        self.assertTrue(np.array_equal(selector.columns, ["var1", "var2", "var3", "var4"]))
        self.assertTrue(self.data.df.equals(self.df))
        self.assertTrue(x.df.equals(self.df))

    def test_univariate_some(self):

        # Learn uniselector
        selector = UniSelector()
        selector.learn(self.data, min_occ=0.21, max_occ=0.79, include_missings=True)

        self.assertTrue(np.array_equal(selector.columns, ["var2", "var4"]))
        self.assertTrue(self.data.df.equals(self.df))

        # Apply not inplace
        x = selector.apply(self.data, inplace=False)
        self.assertTrue(self.data.df.equals(self.df))
        self.assertTrue(np.array_equal(x.df.columns, ["var2", "var4"]))

        # Apply inplace
        selector.apply(self.data, inplace=True)
        self.assertTrue(np.array_equal(self.data.df.columns, ["var2", "var4"]))

    def test_univariate_none(self):

        # Learn uniselector
        selector = UniSelector()
        selector.learn(self.data, min_occ=1, max_occ=0.1)

        self.assertTrue(np.array_equal(selector.columns, []))
        self.assertTrue(self.data.df.equals(self.df))

        # Apply not inplace
        x = selector.apply(self.data, inplace=False)
        self.assertTrue(self.data.df.equals(self.df))
        self.assertTrue(np.array_equal(x.df.columns, []))

        # Apply inplace
        selector.apply(self.data, inplace=True)
        self.assertTrue(np.array_equal(self.data.df.columns, []))

    def test_univariate_always(self):

        # Learn uniselector
        selector = UniSelector()
        selector.learn(self.data, min_occ=0.21, max_occ=0.79, include_missings=True, set_aside=["var1"])

        self.assertTrue(np.array_equal(selector.columns, ["var1","var2", "var4"]))
        self.assertTrue(self.data.df.equals(self.df))

        # Apply not inplace
        x = selector.apply(self.data, inplace=False)
        self.assertTrue(self.data.df.equals(self.df))
        self.assertTrue(np.array_equal(x.df.columns, ["var1","var2", "var4"]))

        # Apply inplace
        selector.apply(self.data, inplace=True)
        self.assertTrue(np.array_equal(self.data.df.columns, ["var1","var2", "var4"]))

    def test_univariate_hard(self):

        # Learn uniselector
        selector = UniSelector()
        selector.learn(self.data, variable_select=["var3", "var4"])

        self.assertTrue(np.array_equal(selector.columns, ["var3", "var4"]))
        self.assertTrue(self.data.df.equals(self.df))

        # Apply not inplace
        x = selector.apply(self.data, inplace=False)
        self.assertTrue(self.data.df.equals(self.df))
        self.assertTrue(np.array_equal(x.df.columns, ["var3", "var4"]))

        # Apply inplace
        selector.apply(self.data, inplace=True)
        self.assertTrue(np.array_equal(self.data.df.columns, ["var3", "var4"]))
