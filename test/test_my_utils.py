import unittest
from my_utils import rename_wav_as_extension

class TestStringMethods(unittest.TestCase):

    def test_alwaysRenameFileAsWav(self):
        self.assertEqual(rename_wav_as_extension('a_bc.mp3'), 'a_bc.wav')
        self.assertEqual(rename_wav_as_extension('ab+c'), 'ab+c.wav')
        self.assertEqual(rename_wav_as_extension('a-bc.flac.mp3'), 'a-bc.flac.wav')

if __name__ == '__main__':
    unittest.main()

