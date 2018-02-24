import sys, os
import upandas as upd

# Run a single Python script
# For many simple, single file projects, you may find it inconvenient
# to write a complete Dockerfile. In such cases, you can run a Python
# script by using the Python Docker image directly:

# versions to consider: 3 (600+ MB), slim (150 MB) alpine (90 MB)
# $ docker run -it --rm --name my-running-script -v "$PWD":/usr/src/myapp -w /usr/src/myapp python:3 python your-daemon-or-script.py
# $ docker run -it --rm -v "$PWD":/usr/src/upandas -w /usr/src/upandas python:alpine python upandas_test.py

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no testing approach supplied, see...')
        sys.exit(1)

    env = sys.argv[1]

    if env == 'local':
        print('Testing locally')
    elif env == 'docker':
        print('Using docker to test')
        ex = os.system(
            'docker run -it --rm -v "$PWD":/usr/src/upandas -w /usr/src/upandas '
            'python:alpine python upandas_test.py local')

        sys.exit(os.WEXITSTATUS(ex))
    elif env == 'virtualenv':
        raise NotImplementedError
    else:
        print('Unsupported environment: {}'.format(env))

    sys.argv = sys.argv[:1]  # strip our settings out

import unittest
import math

skip_pandas_tests = True  # TODO: make this explicit in the sys.argv stuff above
try:
    import pandas as pd
    skip_pandas_tests = False
except:
    pass

# Series methods
# ==============


class TestSeriesInit(unittest.TestCase):
    # dict, list, single value, another series, iterator
    def test_basic_init(self):
        samples = [[1, 2, 3], [4, 5, 6],
                   list(range(1000)), [1, None, 2, None], []]

        for ds in samples:
            s = upd.Series(ds)
            self.assertEqual(len(s), len(ds))

            # test shapes
            self.assertEqual(len(s), s.shape[0])
            self.assertEqual(len(s.shape), 1)
            self.assertEqual(type(s.shape), tuple)

            for j, el in enumerate(s):
                self.assertEqual(el, ds[j])

            if not skip_pandas_tests:
                pass
                # TODO: add a function to compare pd.Series and upd.Series
                # spd = pd.Series(ds)
                # self.assertEqual([j for j in s], [j for j in spd])


class TestSeriesApply(unittest.TestCase):
    # TODO: args, kwargs?
    def test_apply(self):
        s = upd.Series([1, 2, 3])
        s = s.apply(lambda x: x**2 - 3)

        self.assertEqual(s.values, [-2, 1, 6])


class TestSeriesCopy(unittest.TestCase):
    def test_copy(self):
        s = upd.Series([1, 2, 3])
        sc = s.copy()
        self.assertEqual(s.values, sc.values)
        sc[0] = 10
        self.assertNotEqual(s.values,
                            sc.values)  # TODO: add comparisons of frames?

    def test_deep_copy(self):  # ...or lack thereof
        s = upd.Series([1, 2, {'foo': 'bar'}])
        sc = s.copy()
        sc[2]['foo'] = 'baz'
        self.assertEqual(s[2]['foo'], sc[2]['foo'])



class TestSeriesValues(unittest.TestCase):
    def test_values(self):
        samples = [[1, 2, 3], [4, 5, 6],
                   list(range(1000)), [1, None, 2, None], []]

        for ds in samples:
            s = upd.Series(ds)
            self.assertEqual(s.values, ds)


if __name__ == '__main__':
    unittest.main()
