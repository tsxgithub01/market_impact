import unittest


class MyTestCase(unittest.TestCase):

    # TODO to be completed
    def test_pipeline(self):
        from model_processing.pipeline import num_pipeline
        import numpy as np
        X = np.array([[1, 2], [np.nan, 3], [7, 6]])
        x1 = num_pipeline.fit_transform(X)
        expected = np.array([[-1.22474487, -0.98058068],
                             [0., -0.39223227],
                             [1.22474487, 1.37281295]])
        self.assertEqual(True, True)

    def test_reg_model(self):
        from model_processing.ml_reg_models import Ml_Reg_Model
        m = Ml_Reg_Model('linear_nnls')
        m.build_model()
        m.train_model([[1.0], [2.0]], [-0.1, -0.2])
        print(m.output_model())
        self.assertEqual(True, True)

    def test_utils(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
