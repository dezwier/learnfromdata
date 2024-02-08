import unittest
import pandas as pd
import numpy as np
from lfd import Data, set_logging


class DataTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.df = pd.DataFrame([
            [1, 2., "3", "a"],
            [4, 8., "6", "b"],
            [3, 5., "5", "b"],
            [0, 1., "2", "b"],
            [7, 8., "9", "c"]
        ], columns=[f"var{i}" for i in range(4)])
        
        # Logging
        set_logging(stdout=False)

    def test_data_constructor(self):

        # Construct
        data = Data(self.df, name="Test_df")

        # Test dataframe        
        self.assertTrue(self.df.equals(data.df))
        self.assertEqual(data.name, "Test_df")
        
        size = 10000
        df = pd.DataFrame({
            "var0": np.random.normal(0, 1, size=size),
            "var1": np.random.choice(["a", "b"], size=size),
            "var2": np.random.randint(100, size=size),
            "var3": np.random.choice(["1", "2"], size=size),
        }, columns=[f"var{i}" for i in range(4)])

        data = Data(df, name="Test_df")

        # Test dataframe        
        self.assertTrue(df.equals(data.df))
        self.assertEqual(data.name, "Test_df")

    def test_data_import(self):
        
        data = Data('datasets/titanic.csv', name='titanic')

        # Test dataframe        
        self.assertEqual(data.name, "titanic")

        # Test df sizes
        self.assertEqual(data.df.shape[0], 891)
        self.assertEqual(data.df.shape[1], 11)

    def test_data_split_select(self):
        
        data = Data(self.df, name="Test_df")

        # Split
        data.split(subset=0, mask=None, test_size=0.4)

        # Select
        train = data.select(subset='Train', drop=True)
        test = data.select(subset='Test', drop=True)

        # Test sizes
        self.assertEqual(train.df.shape[0], 3)
        self.assertEqual(test.df.shape[0], 2)
        self.assertEqual(data.df.shape[0], 0)
