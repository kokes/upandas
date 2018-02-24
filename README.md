**tl;dr: When trying to replicate a complex project, don't think it's just about recreating the visible API and calling it a day. Most of the logic is underneath, in unexported fields and methods, in inherited logic and careful optimisations. But hey, it's fun even when you fail miserably.**

We all have some pet projects that we hope to finish some day. This one was a bit odd in that I kinda knew that I would fail eventually. But I had tons of fun working on this anyway. This is a sort of *lessons learned* document.

The premise was fairly innocuous - I would work with [pandas](https://pandas.pydata.org/) on a daily basis for many years, but its fairly heavyweight nature would bite me from time to time. While it's trivial to install on a desktop, it would sometimes be problematic on remote servers, in minimal Docker images, on ARM etc. I wondered if there was a way to overcome this.

Now, pandas is not really that heavyweight. The number of required dependencies is fairly short, with `numpy` being the biggest one (and fairly essential one, as it is used for storage). But once you replace the storage engine, you can solve most of the remaining problems with pure Python (pandas uses Cython in quite a few places for performance reasons), which is fairly well equipped, we're talking `csv`, `json`, `urllib`, `itertools`, `gzip` and other goodies in the standard library. And me being quite a fan of pure Python solutions, I decided to give this **"pandas in pure Python and, what the heck, in a single file"** project a go.

The idea was that you'd `curl` a single file from GitHub and off you'd go. I imagined that a small subset of pandas' API would serve one well. Now let's see what we're up against:

```
$ git clone --depth=1 https://github.com/pandas-dev/pandas
$ loc pandas
--------------------------------------------------------------------------------
 Language             Files        Lines        Blank      Comment         Code
--------------------------------------------------------------------------------
 Python                 589       332060        63990        21213       246857
 reStructuredText        45        39548        11076            0        28472
 HTML                    16        18977         2323          201        16453
 Plain Text              49        17088         4879            0        12209
 C                       11        12234         1720         1223         9291
 Autoconf                13         5206          974          250         3982
 C/C++ Header            28         5293          845         1033         3415
 Bourne Shell            32          974          230          112          632
 Markdown                 6          352           72            0          280
 YAML                     8          372           51           43          278
 JSON                     3          128           15            0          113
 Batch                    4          103            6            0           97
 INI                      1           82           11            0           71
 Makefile                 2           38           10            0           28
 Toml                     1            9            0            0            9
 ASP.NET                  2            2            0            0            2
--------------------------------------------------------------------------------
 Total                  810       432466        86202        24075       322189
--------------------------------------------------------------------------------
```

Yikes, luckily we're not rebuilding all of pandas, just some of its features. It took me maybe a couple dozen hours to get something together, here's what I learned.

### Don't start with the high level stuff

How do people get data into pandas? Quite a few use `read_csv`... so I started with that (notice it's no longer in the code here). After building it for a few hours, adding compression support, HTTP handling, tests, ... I realised I didn't have any data structures to save the data into. I hadn't built any DataFrames or Series. This was a very early error.

### If you're recreating the visible API, you will fail

This is a big one.

I built a constructor for a `Series` object and then a few methods. After implementing my favourite ones, I started remembering what other ones there were. So I went to the documentation and felt like parsing it... then it dawned on me - I can use `dir`!

The premise was that you create an object and then call `dir` on it to get all the methods it implements. So I'd do things like `pd.Series(list('abc')).str.__dir__()` or `pd.to_datetime(pd.Series(['2017-11-20'])).dt.__dir__()` and I'd get all the methods that needed implementing. But that turned out to be a very wrong approach.

If you try and recreate everything that's visible, you will fail to capture the essence of productive programming - inheritance (or reuse, depending on context). A whole host of methods were meant to be implemented for `Series` (or even `Index`!) and then reused in `DataFrame`s and I wasn't capturing this logic.

Pandas' API surface isn't giant because there are a lot of methods implemented (well, there are, but way fewer than one might think), but mainly because it leverages commonalities between different objects that share some properties.

### Python's STL is amazing

OK, I already knew that going into this, but working with it only reaffirmed this simple fact. I already listed some of the libraries above, but then there is `unittest`, `datetime`, the underused `functools` etc. Whenever I teach someone Python, I urge them to give the standard library a go first. You get great mileage out of that built in beast and you spare yourself a world of pain when porting your code or maitaining it as your random libraries change their APIs willy nilly.

### There are more magic methods than I knew

I knew a few, like `__getitem__` or `__len__`, but I got to know about love `__repr__`, a suite of algebraic methods (`__add__`, `__div__` and the like), and odd ones like `__invert__` or `__contains__`.

Utilising magic methods allows for hiding a lot of logic that would otherwise go into named methods and awkward APIs. Just imagine having to do `df.add(dd)` instead of plain `df + dd`, which is fairly easy to implement. It goes beyond just `+-/*`, you can implement `__lt__`, `__gt__` etc. to evaluate comparisons, methods for bit shifting, binary operators, there's `__del__` to handle object deletion - e.g. upon garbage collection when a variable is redefined or simply when a program exits (though be careful, builtins are deleted before `__del__` gets triggered, so you may want to use `__exit__` in a `with` block instead).

Anywho, magic methods are excellent and you should learn more about them, see e.g. [this epic post](https://rszalski.github.io/magicmethods/).

### Hard to view, copies in most places

Being quite used to manage some of my memory in Go, be it by pointers or slices (sort of views of underlying arrays). This is done in pandas by utilising numpy storage, which, in turns, uses dense byte arrays and views over chunks of them.

Sadly, because I was working on a pure Python solution, I could mimick this only by using [`bytearray`](https://docs.python.org/3.1/library/functions.html#bytearray), but that would require a lot of code around it to coerce data back and forth. In the end, I decided to go the memory inefficient, but very usable and friendly way - by just using lists of Python primitives.

### Composability is key

While it was not trivial to get the first iteration out the door, I was amazed by how much I could add by writing very little code. I had an internal `_apply` method on Series, which I would then feed a bunch of Python's built-in string methods to generate much of pandas' `.str` API. Same goes for `.dt` and the `datetime` library.

### Purity calls for cautious testing

Building something using just Python's standard library, I had to ensure that the code was not in any way inferered with and my age old desktop installation of Python was the obvious culprit. A simple solution would be a virtual environment, but, for some reason, I have never been a fan. Wanting a bit more isolation without much work, I opted for Docker. Sure, it's a heavy dependency, but it was only optional. I built the `_test.py` file to be a self-contained program, which allowed for argument to be passed, so you could just test the thing using your existing installation or in Docker (and virtual environments were planned).

I opted for one of the simplest distros, I went with `alpine`. By default, it weighs just a couple megs, when bundled with Python, it stops somewhere shy of 100 megs. Not great, but you can't get it much lower than that.

The resulting command was very basic. Just run the base `python:alpine` image, inject my code into it and run the `_test.py` file and they self destruct. Easy peasy and it runs in seconds.

```
docker run -it --rm -v "$PWD":/usr/src/upandas -w /usr/src/upandas python:alpine python upandas_test.py local
```

### Other tidbits

- [devdocs.io](http://devdocs.io/) are great, they are a (blazing fast) source of documentation that I reviewed constantly.
- Custom exceptions are cool. Sure, I used a lot of `NotImplementedError`, but then I wrapped this around into `WontImplement` to signify some parts of the API there just not possible to recreate. In general, I prefer `NotImplementedError` over a bunch of TODOs, because at least the program crashes.
- Chaining is awesome. Every developer using pandas knows this, you can call things like `pd.read_csv('...').loc[...].head().T.drop(...)` and you can just keep going. This can be easily implemented by just returning the object back upon each method call.
