import unittest
from math import exp, log, sqrt

from logreg import LogReg, Example
from numpy import zeros

kTOY_VOCAB = "BIAS_CONSTANT A B C D".split()
kPOS = Example(1, "A:4 B:3 C:1".split(), kTOY_VOCAB, None)
kNEG = Example(0, "B:1 C:3 D:4".split(), kTOY_VOCAB, None)

class TestKnn(unittest.TestCase):
    def setUp(self):
        self.logreg_unreg = LogReg(5, 0.0, lambda x: 1.0)
        self.logreg_reg = LogReg(5, 0.25, lambda x: 1.0)

    def test_unreg(self):
        print(self.logreg_unreg.beta)
        print(kPOS.x)
        beta = self.logreg_unreg.sg_update(kPOS, 1)
        self.assertAlmostEqual(beta[0], .5)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(beta[2], 1.5)
        self.assertAlmostEqual(beta[3], 0.5)
        self.assertAlmostEqual(beta[4], 0.0)

        print(self.logreg_unreg.beta)
        print(kPOS.x)
        beta = self.logreg_unreg.sg_update(kNEG, 2)
        self.assertAlmostEqual(beta[0], -0.47068776924864364)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(beta[2], 0.5293122307513564)
        self.assertAlmostEqual(beta[3], -2.4120633077459308)
        self.assertAlmostEqual(beta[4], -3.8827510769945746)

    def test_tfidf_features(self):
        vocab = ["BIAS_CONSTANT", "A", "B", "C"]
        example = Example(1, ["A:2", "B:1"], vocab, [0, 1, 3, 0], 3)
        a = (1 + log(2)) * (1 + log(4 / 2))
        b = 1.0
        length = sqrt(a * a + b * b)
        self.assertEqual(example.x[1], 2)
        self.assertEqual(example.x_tfidf[0], 1)
        self.assertAlmostEqual(example.x_tfidf[1], a / length)
        self.assertAlmostEqual(example.x_tfidf[2], b / length)
        self.assertEqual(example.x_tfidf[3], 0)

        empty = Example(0, ["unknown:2"], vocab, [0, 1, 3, 0], 3)
        self.assertEqual(empty.x_tfidf[0], 1)
        for x in range(1, len(vocab)):
            self.assertEqual(empty.x_tfidf[x], 0)

    def test_tfidf_update_and_evaluation(self):
        vocab = ["BIAS_CONSTANT", "A", "B"]
        example = Example(1, ["A:3"], vocab, [0, 1, 1], 2)
        model = LogReg(3, 0.0, lambda t: 1.0)
        beta = model.sg_update(example, 1, use_tfidf=True)
        self.assertAlmostEqual(beta[0], 0.5)
        self.assertAlmostEqual(beta[1], 0.5)
        self.assertEqual(beta[2], 0)

        # Raw counts predict positive here, but TF-IDF predicts negative.
        model.beta[0] = -2
        model.beta[1] = 1
        logprob, accuracy = model.progress([example], use_tfidf=True)
        self.assertAlmostEqual(logprob, -log(1 + exp(1)))
        self.assertEqual(accuracy, 0)
        raw_logprob, raw_accuracy = model.progress([example])
        self.assertEqual(raw_accuracy, 1)

if __name__ == '__main__':
    unittest.main()
