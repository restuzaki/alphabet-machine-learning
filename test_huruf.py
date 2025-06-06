import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class testHuruf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Membaca data
        url = 'C:/Users/damai/alphabet_machine_Learning/data/letter-recognition.data'
        columns = ['letter', 'x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar',
                   'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
        df = pd.read_csv(url, header=None, names=columns)
        cls.X = df.drop('letter', axis=1)
        cls.y = df['letter']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42)
        cls.model = RandomForestClassifier(n_estimators=100, random_state=42)
        cls.model.fit(cls.X_train, cls.y_train)

    def test_model_accuracy(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.85, "Akurasi model kurang dari 85%")

    def test_prediction_output(self):
        sample = self.X_test.iloc[0].values.reshape(1, -1)
        prediction = self.model.predict(sample)
        self.assertIn(prediction[0], list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                      "Prediksi bukan huruf kapital A-Z")

    def test_model_not_none(self):
        self.assertIsNotNone(self.model, "Model tidak boleh None")

if __name__ == '__main__':
    unittest.main()
