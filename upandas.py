__version__ = '0.0.1'


class WontImplement(NotImplementedError):
    def __init__(self, msg=None):
        if msg is None:
            msg = 'This feature won\'t be implemented in upandas'

        super().__init__(msg)


class dtype:
    def __init__(self):
        pass


class Iterable:
    """
    For one-dimensional iterables
    """

    # set(dir(s.index)).intersection(dir(s))
    # done: T, transpose, values, min, max, notna, notnull, unique, nunique, hasnans,
    # fillna, values, isna, isnull, copy, empty, name, map, shape, tolist, data, dtype,
    # isin, dropna, all, any, is_unique

    # remaining: (notable: groupby, value_counts, astype, size, ndim, sort_values)
    # append, argmax, argmin, argsort, asof, astype, base,
    # drop, drop_duplicates, duplicated, equals, factorize,
    # flags, get_values, groupby, is_monotonic,
    # is_monotonic_decreasing, is_monotonic_increasing,
    # item, itemsize, memory_usage, nbytes, ndim,
    # ravel, reindex, rename, repeat, searchsorted, shift, size,
    # sort_values, strides, take, to_frame, value_counts,
    # view, where

    @property
    def str(self):
        return StringMethods(self)

    @property
    def data(self):
        return self.values

    @property
    def dt(self):
        return DateMethods(self)

    @property
    def dtype(self):
        return self._dtype

    @property
    def empty(self):
        return len(self) == 0

    @property
    def is_unique(self):
        return self.nunique() == len(self)

    @property
    def hasnans(self):
        for el in self:
            if isna(el):
                return True

        return False

    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = None
        return self._name

    @name.setter
    def name(self, val):
        # TODO: check it's hashable
        self._name = val

    @property
    def shape(self):
        return (len(self), )

    @property
    def T(self):
        if isinstance(self, (Series, Index)):
            return self

        raise NotImplementedError

    @property
    def values(self):
        return self._data

    def __init__(self, data):
        if isinstance(data, list):
            self._data = data.copy()
        elif isinstance(data, (int, float, bool, str)):
            raise NotImplementedError
        else:
            raise TypeError('unexpected input data')

    def __iter__(self):
        for el in self._data:
            yield el

    def __getitem__(self, arg):
        if type(arg) is int:
            if arg < 0 or arg >= len(self):
                raise ValueError(arg)  # TODO: coat it in some text

            return self._data[arg]
        elif type(arg) is slice:
            return self.iloc[arg]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self._data)

    def all(self):
        for el in self:
            if el == False:
                return False

        return True

    def any(self):
        for el in self:
            if el == True:
                return True

        return False

    def astype(self, dtype):
        raise NotImplementedError

    def copy(self):
        # TODO: all other metadata?
        cp = object.__new__(self.__class__)
        cp.__init__(self._data)

        return cp

    def dropna(self):
        s = self.copy()
        s._data = [j for j in self if notna(j)] # TODO: too low level
        return s

    def fillna(self, value=None, downcast=None):
        if downcast is not None:
            raise NotImplementedError

        if not self.hasnans:
            return self  # should be a shallow copy instead

        return self.map(lambda x: x if notna(x) else value)


    def isna(self):
        return [isna(j) for j in self]

    def isnull(self):
        return self.isna()

    def isin(self, values):
        if not isinstance(values, (set, list)):
            raise TypeError('values supplied must be a list or a set')

        if type(values) is set:
            vset = values
        else:
            vset = set(values)

        return self.map(lambda x: x in vset)

    def map(self, func):
        # TODO: more parameters?
        s = self.copy()
        s._data = [func(j) for j in self] # TODO: too low level
        # TODO: too low level

        return s

    def max(self):
        from functools import reduce
        return reduce(lambda a, b: a if a > b else b, self)

    def min(self):
        from functools import reduce
        return reduce(lambda a, b: a if a < b else b, self)

    def notna(self):
        if hasattr(self, 'apply'):
            return self.apply(notna)

        return [notna(j) for j in self]

    def notnull(self):
        return self.notna()

    def nunique(self):
        uels = set()
        for el in self:
            uels.add(el)

        return len(uels)

    def transpose(self):
        return self.T

    def tolist(self):
        return self.values.copy()

    def unique(self):
        uels = set()
        for el in self:
            uels.add(el)

        vals = sorted(list(uels))

        # TODO: can't we just call __init__?
        if isinstance(self, Series):
            return Series(vals)
        elif isinstance(self, Index):
            return Index(vals)
        else:
            raise NotImplementedError


class Index(Iterable):
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.html
    # set(dir(s.index)).difference(dir(s))
    # also define (inherited):
    # - RangeIndex, Int64Index, UInt64Index, Float64Index, CategoricalIndex,
    #   IntervalIndex, MultiIndex, DatetimeIndex, TimedeltaIndex, PeriodIndex
    pass


class LocIndexer():
    def __init__(self, s):
        pass


class iLocIndexer():
    def __init__(self, s):
        self.s = s

    def __getitem__(self, arg):
        if type(arg) is int:
            return self.s[arg]

        if type(arg) is slice:
            if arg.step is not None:
                raise NotImplementedError

            # TODO: can we just give a view instead? That's the pandas way
            return Series(
                self.s._data[arg.start:arg.stop],
                index=self.s._index[arg.start:arg.stop])

        raise NotImplementedError(type(arg))


class Frame:
    # TODO: do we need a generic class here, or can we move it under Series?
    #       check set(dir(s)).intersection(dir(df))
    #
    # set(dir(s)).difference(dir(s.index) for extra methods on top of Iterables

    @property
    def iloc(self):
        return iLocIndexer(self)

    @property
    def loc(self):
        return LocIndexer(self)

    def apply(self, func, *args, **kwargs):
        if not hasattr(func, '__call__'):
            raise TypeError('apply method needs to be callable')

        # TODO: dtype may have changed (relates to convert_dtype arg, which
        # defaults to True)
        res = []
        for el in self:
            res.append(func(el, *args, **kwargs))

        # TODO: what about the index?
        return Series(res)

    def between(self, left, right, inclusive=True):
        if inclusive:
            return self.apply(
                lambda x: (x >= left) and (x <= right) if notna(x) else None)

    def count(self, level=None):
        import math
        if level is not None:
            raise NotImplementedError  # MultiIndex not supported

        # TODO: allocates quite a bit (but it is correct)
        return len(self.notnull())

    def filter(self, items=None, like=None, regex=None, axis=None):
        raise NotImplementedError


class Series(Iterable, Frame):
    def __init__(self, data=None, index=None):
        # if not hasattr(data, '__iter__'):
        #     raise ValueError('cannot iterate through data')
        self._index = []
        self._data = []
        self._dtype = object

        if data is None:
            self._index = [] if index is None else index
            self._data = [None] * len(self._index)
        elif isinstance(data, dict):
            if index is not None:
                for el in index:
                    self._index.append(el)
                    self._data.append(data.get(el, None))
            else:
                for k, v in data.items():
                    self._index.append(k)
                    self._data.append(v)
        elif isinstance(data, list):
            self._data = data.copy()
            if index is not None:
                if len(index) != len(data):
                    raise ValueError(
                        'expecting supplied index to be the same length as data'
                    )
                self._index = index
            else:
                # TODO: does this need to be materialised?
                # it should be a separate class
                self._index = range(len(data))
        elif isinstance(data, (int, float, bool, str)):
            raise NotImplementedError
        elif isinstance(data, Series):
            raise NotImplementedError
        else:
            raise TypeError('unexpected input data')

    def __repr__(self):
        maxind = 0
        maxval = 0

        tr = 30  # first n rows and last n rows to display

        if len(self) <= 2 * tr:
            inds = list(range(len(self)))
        else:
            inds = list(range(tr)) + list(range(-tr, 0))

        for j in inds:
            # TODO: repr length is not considered (escaped newlines are longer than explicit)
            ln = len(str(self._index[j]))
            maxind = ln if ln > maxind else maxind

            ln = len(str(self._data[j]))
            maxval = ln if ln > maxval else maxval

        ret = []
        for j in inds:
            # repr()[1:-1] better than foo.encode('utf8').decode('unicode_escape')?
            # it's primarily because of newlines (so do we just replace them?)
            # repr(str) is here just to coerce non-string values - could be eliminated
            # with an if statement
            ret.append('%s %s' %
                       (repr(str(self._index[j]))[1:-1].ljust(maxind),
                        repr(str(self._data[j]))[1:-1].rjust(maxval)))

            if len(self) > 2 * tr and j == tr - 1:
                ret.append(' ' * (maxind + 1) + '...')

        if len(self) > 2 * tr:
            ret.append('Length: %d, dtype: %s' % (len(self), self.dtype))
        else:
            ret.append('dtype: %s' % self.dtype)

        return '\n'.join(ret)

    def __setitem__(self, arg, value):
        # TODO: dtype may be affected
        if type(arg) != int:
            raise NotImplementedError
        if arg < 0 or arg >= len(self):
            raise ValueError(arg)  # TODO: coat it in some text

        self._data[arg] = value

    # a common method to handle __add__, __div__, __mul__ etc.
    def _algebra(self, method):
        if method not in ['__add__']:
            raise NotImplementedError

        if not isinstance(other, Series):
            # print(type(other))
            raise NotImplementedError

        if len(self) != len(other):
            raise ValueError('cannot combine two Series of unequal shapes')

        # careful about indices
        ind = set(self._index)
        othind = set(other._index)
        allind = ind.union(othind)

        # no common index values
        # TODO: test:
        # s = upd.Series([1,2,3], index=list('abc'))
        # s2 = upd.Series([1,2,3], index=list('def'))
        # assertEqual(s+s2, pd.Series([None]*6, index=list('abcdef')))

        if len(ind.intersection(othind)) == 0:
            return Series(index=sorted(list(allind)))

        raise NotImplementedError

    def __add__(self, other):
        import inspect
        method = inspect.currentframe().f_code.co_name
        return self._algebra(method)

    def __radd__(self, other):
        raise NotImplementedError

    def __div__(self, other):
        raise NotImplementedError

    def __invert__(self):
        return self.apply(lambda x: not (x) if type(x) is bool else ~x)


class StringMethods:
    """
    String methods that are mostly 1:1 calls to built-in string methods in Python
    """

    def __init__(self, s):
        if s.dtype != object:
            raise TypeError(
                'can only use .str methods for columns with dtype `object`')
        self.s = s

    # TODO: these should work on indices as well... how? (e.g. lower/upper)

    def _apply(self, func):
        """
        Methods here usually work just on strings, so we need to type check.
        This bit of boilerplate will help us do that.

        You can supply a function or a string. Said function will be applied
        to all string elements. If a string is supplied, a method on said string
        will be invoked.
        """
        if type(func) is str:
            return self._apply(lambda x: getattr(x, func)())

        return self.s.apply(lambda x: func(x) if type(x) is str else None)

    def capitalize(self):
        return self._apply(lambda x: x.capitalize())

    def cat(self):
        raise NotImplementedError

    def center(self):
        raise NotImplementedError

    def contains(self, needle):
        # TODO: flags and another params
        return self._apply(lambda x: needle in x)

    def count(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

    def endswith(self):
        return self._apply('endswith')

    def extract(self):
        raise NotImplementedError

    def extractall(self):
        raise NotImplementedError

    def find(self):
        raise NotImplementedError

    def findall(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def get_dummies(self):
        raise NotImplementedError

    def index(self):
        raise NotImplementedError

    def isalnum(self):
        return self._apply('isalnum')

    def isalpha(self):
        return self._apply('isalpha')

    def isdecimal(self):
        return self._apply('isdecimal')

    def isdigit(self):
        return self._apply('isdigit')

    def islower(self):
        return self._apply('islower')

    def isnumeric(self):
        return self._apply('isnumeric')

    def isspace(self):
        return self._apply('isspace')

    def istitle(self):
        return self._apply('istitle')

    def isupper(self):
        return self._apply('isupper')

    def join(self, sep):
        return self._apply(lambda x: sep.join(x))

    def len(self):
        return self._apply('__len__')

    def ljust(self, width, fillchar=' '):
        return self._apply(lambda x: x.ljust(width, fillchar))

    def lower(self):
        return self._apply('lower')

    def lstrip(self):
        return self._apply('lstrip')

    def match(self):
        raise NotImplementedError

    def normalize(self):
        raise NotImplementedError

    def pad(self):
        raise NotImplementedError

    def partition(self):
        # returns a DataFrame, TBA
        raise NotImplementedError

    def repeat(self):
        raise NotImplementedError

    def replace(self):
        raise NotImplementedError

    def rfind(self):
        raise NotImplementedError

    def rindex(self):
        raise NotImplementedError

    def rjust(self, width, fillchar=' '):
        return self._apply(lambda x: x.rjust(width, fillchar))

    def rpartition(self):
        raise NotImplementedError

    def rsplit(self):
        raise NotImplementedError

    def rstrip(self):
        return self._apply('rstrip')

    def slice(self):
        raise NotImplementedError

    def slice_replace(self):
        raise NotImplementedError

    def split(self):
        raise NotImplementedError

    def startswith(self):
        raise NotImplementedError

    def strip(self):
        return self._apply('strip')

    def swapcase(self):
        raise NotImplementedError

    def title(self):
        return self._apply('title')

    def translate(self):
        raise NotImplementedError

    def upper(self):
        return self._apply('upper')

    def wrap(self, width, **kwargs):
        import textwrap
        tw = textwrap.TextWrapper(width, **kwargs)
        return self._apply(lambda x: tw.fill(x) if type(x) is str else None)

    def zfill(self, width):
        return self.rjust(width, fillchar='0')


class DateMethods:
    def __init__(self, s):
        if s.dtype == 'datetime':  # TODO: once we specify types
            raise TypeError(
                'can only use .dt methods for columns with dtype `datetime`')
        self.s = s

    def _strftime_helper(self, format):
        pass

    def _ts_helper(self, property):
        from datetime import datetime
        # some are callable (date), some aren't (minute)
        if property in ['date', 'time', 'weekday']:
            return self.s.apply(
                lambda x: getattr(datetime.fromtimestamp(x), property)())

        return self.s.apply(
            lambda x: getattr(datetime.fromtimestamp(x), property))

    def ceil(self):
        raise NotImplementedError

    def date(self):
        return self._ts_helper('date')

    def day(self):
        return self._ts_helper('day')

    def dayofweek(self):
        return self._ts_helper('weekday')

    def dayofyear(self):
        raise NotImplementedError

    def days_in_month(self):
        raise NotImplementedError

    def daysinmonth(self):
        raise NotImplementedError

    def floor(self):
        raise NotImplementedError

    def freq(self):
        raise NotImplementedError

    def hour(self):
        return self._ts_helper('hour')

    def is_leap_year(self):
        raise NotImplementedError

    def is_month_end(self):
        raise NotImplementedError

    def is_month_start(self):
        raise NotImplementedError

    def is_quarter_end(self):
        raise NotImplementedError

    def is_quarter_start(self):
        raise NotImplementedError

    def is_year_end(self):
        raise NotImplementedError

    def is_year_start(self):
        raise NotImplementedError

    def microsecond(self):
        return self._ts_helper('microsecond')

    def minute(self):
        return self._ts_helper('minute')

    def month(self):
        return self._ts_helper('month')

    def nanosecond(self):
        raise NotImplementedError

    def normalize(self):
        raise NotImplementedError

    def quarter(self):
        raise NotImplementedError

    def round(self):
        raise NotImplementedError

    def second(self):
        return self._ts_helper('second')

    def strftime(self):
        raise NotImplementedError

    def time(self):
        return self._ts_helper('time')

    def to_period(self):
        raise NotImplementedError

    def to_pydatetime(self):
        raise NotImplementedError

    def tz(self):
        raise NotImplementedError

    def tz_convert(self):
        raise NotImplementedError

    def tz_localize(self):
        raise NotImplementedError

    def week(self):
        raise NotImplementedError

    def weekday(self):
        return self._ts_helper('weekday')

    def weekday_name(self):
        from datetime import datetime
        wdays = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
            'Sunday'
        ]
        return self.s.apply(
            lambda x: wdays[datetime.fromtimestamp(x).weekday()])

    def weekofyear(self):
        raise NotImplementedError

    def year(self):
        return self._ts_helper('year')


class DataFrame:
    """
    T, abs, add, add_prefix, add_suffix, agg, aggregate, align, all, any, append,
    apply, applymap, as_matrix, asfreq, asof, assign, astype, at, at_time, axes,
    between_time, bfill, bool, boxplot, clip, clip_lower, clip_upper, columns,
    combine, combine_first, compound, copy, corr, corrwith, count, cov, cummax,
    cummin, cumprod, cumsum, describe, diff, div, divide, dot, drop,
    drop_duplicates, dropna, dtypes, duplicated, empty, eq, equals, eval, ewm,
    expanding, ffill, fillna, filter, first, first_valid_index, floordiv, from_dict,
    from_items, from_records, ftypes, ge, get, get_dtype_counts, get_ftype_counts,
    get_values, groupby, gt, head, hist, iat, idxmax, idxmin, iloc, index,
    infer_objects, info, insert, interpolate, is_copy, isin, isna, isnull, items,
    iteritems, iterrows, itertuples, ix, join, keys, kurt, kurtosis, last,
    last_valid_index, le, loc, lookup, lt, mad, mask, max, mean, median, melt,
    memory_usage, merge, min, mod, mode, mul, multiply, ndim, ne, nlargest, notna,
    notnull, nsmallest, nunique, pct_change, pipe, pivot, pivot_table, plot, pop,
    pow, prod, product, quantile, query, radd, rank, rdiv, reindex, reindex_axis,
    reindex_like, rename, rename_axis, reorder_levels, replace, resample,
    reset_index, rfloordiv, rmod, rmul, rolling, round, rpow, rsub, rtruediv,
    sample, select, select_dtypes, sem, set_axis, set_index, shape, shift, size,
    skew, slice_shift, sort_index, sort_values, squeeze, stack, std, style, sub,
    subtract, sum, swapaxes, swaplevel, tail, take, to_clipboard, to_csv, to_dense,
    to_dict, to_excel, to_feather, to_gbq, to_hdf, to_html, to_json, to_latex,
    to_msgpack, to_panel, to_parquet, to_period, to_pickle, to_records, to_sparse,
    to_sql, to_stata, to_string, to_timestamp, to_xarray, transform, transpose,
    truediv, truncate, tshift, tz_convert, tz_localize, unstack, update, values,
    var, where, xs"""
    pass


"""
Pandas functions (some may be outdated, see e.g. https://github.com/pandas-dev/pandas/issues/13790)

$ import textwrap
$ textwrap.fill(', '.join([j for j in dir(pd) if not j.startswith('_')]), 80)

"""

# Categorical, CategoricalIndex, DataFrame, DateOffset, DatetimeIndex, ExcelFile,
# ExcelWriter, Expr, Float64Index, Grouper, HDFStore, Index, IndexSlice,
# Int64Index, Interval, IntervalIndex, MultiIndex, NaT, Panel, Panel4D, Period,
# PeriodIndex, RangeIndex, Series, SparseArray, SparseDataFrame, SparseList,
# SparseSeries, Term, TimeGrouper, Timedelta, TimedeltaIndex, Timestamp,
# UInt64Index, WidePanel, api, bdate_range, compat, concat, core, crosstab, cut,
# date_range, datetime, datetools, describe_option, errors, eval, ewma, ewmcorr,
# ewmcov, ewmstd, ewmvar, ewmvol, expanding_apply, expanding_corr,
# expanding_count, expanding_cov, expanding_kurt, expanding_max, expanding_mean,
# expanding_median, expanding_min, expanding_quantile, expanding_skew,
# expanding_std, expanding_sum, expanding_var, factorize, get_dummies, get_option,
# get_store, groupby, infer_freq, interval_range, io, isna, isnull, json, lib,
# lreshape, match, melt, merge, merge_asof, merge_ordered, notna, notnull, np,
# offsets, option_context, options, ordered_merge, pandas, parser, period_range,
# pivot, pivot_table, plot_params, plotting, pnow, qcut, read_clipboard, read_csv,
# read_excel, read_feather, read_fwf, read_gbq, read_hdf, read_html, read_json,
# read_msgpack, read_parquet, read_pickle, read_sas, read_sql, read_sql_query,
# read_sql_table, read_stata, read_table, reset_option, rolling_apply,
# rolling_corr, rolling_count, rolling_cov, rolling_kurt, rolling_max,
# rolling_mean, rolling_median, rolling_min, rolling_quantile, rolling_skew,
# rolling_std, rolling_sum, rolling_var, rolling_window, scatter_matrix,
# set_eng_float_format, set_option, show_versions, stats, test, testing,
# timedelta_range, to_datetime, to_msgpack, to_numeric, to_pickle, to_timedelta,
# tools, tseries, tslib, unique, util, value_counts, wide_to_long


def isna(value):
    if hasattr(value, '__iter__'):
        return [isna(j) for j in value]

    import math
    return (value is None) or math.isnan(value)


def notna(value):
    if hasattr(value, '__iter__'):
        return [notna(j) for j in value]

    return not isna(value)


if __name__ == '__main__':
    pass
