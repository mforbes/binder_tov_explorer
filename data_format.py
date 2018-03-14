"""Reference implementation of the Data Format.

To Do:

* There is a bug in the repr code of the _classes.py file.  New-lines are not
  properly implemented.
* Systematically deal with units: i.e. provide a way of taking a units
  dictionary and returning all the the data appropriately scaled.
"""
from collections import OrderedDict
import datetime
import logging
import os.path
import shutil
import subprocess
import textwrap
import time

import numpy as np

_SENTINEL = "_this_dir_is_an_mmfutils_DataSet"
_DOCSTRING = """{info}

------------------------------------------------------
WARNING: This is a generated dataset - edit with care.

Generated on: {now}

Binary data stored in the NPY format described here:

https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
"""


def allclose(a, b, **kw):
    """Return `True` if a and be are close.

    Like np.allclose, but first tries a strict equality test.
    """
    try:
        if a == b:
            return True
    except ValueError:
        pass

    try:
        if np.allclose(a, b, **kw):
            return True
    except TypeError:
        pass
    
    return False
    

def multiline_string_repr(string):
    """Return a representation of the string using multi-line string format
    if possible."""
    if '"""' not in string:
        string = '"""{}\n"""\n'.format(string)
    elif "'''" not in string:
        string = "'''{}\n'''\n".format(string)
    else:
        string = repr(string) + "\n"
    return string


def fill_rep(repr, name="", indent=" "*8):
    indent = indent + " "*len(name + "=[")
    wrapper = textwrap.TextWrapper(
        width=80,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False)
    return wrapper.fill(repr)[len(indent):]


class Array(object):
    """Represents an array stored on disk.

    Attributes
    ----------
    array : array-like or str
       Actual data array or location of data file.
    dims : [str]
       List of dimension names describing each dimension of the array.  The
       actual Dimension objects are obtained from the `_dims` dictionary in
       the model.
    coords : None, {key: arraylike}
       Optional dict of coordinate values associated with each dimension.  If
       not provided, then the coords will be obtained from the Dimension in
       the module dictionary.  This can be used if the specified array is
       tabulated at a different set of abscissa.

    Examples
    ========
    >>> Array('stars.npy', unit=(1, 0), dims=['neutron_star_observable', 'Pc'])
    Array(
        'stars.npy',
        dims=['neutron_star_observable', 'Pc'],
        unit=(1, 0))
    """
    def __init__(self, array, dims=None, unit='1', coords=None):
        if isinstance(array, str):
            self.filename = array
        else:
            self._data = array
        self.unit = unit
        self.dims = dims

    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = np.load(self.filename)
        return self._data

    def __repr__(self, indent="    ", use_numpy=False):
        return (("{}(\n" + indent).format(self.__class__.__name__)
                + (",\n" + indent).join(
                    [repr(self.filename)]
                    +
                    ["{}={}".format(attr, fill_rep(repr(getattr(self, attr)),
                                                   attr,
                                                   indent=" "*4))
                     for attr in sorted(self.__dict__)
                     if attr not in set(['_data', 'filename'])])
                + ")")


class Dimension(object):
    def __init__(self, coords):
        if hasattr(coords, 'tolist'):
            coords = coords.tolist()
        self.coords = coords
        assert coords == eval(repr(coords))

    def __eq__(self, other):
        """Return `True` if the two dimensions are equal."""
        if set(self.__dict__) != set(other.__dict__):
            return False

        return all(allclose(getattr(self, key), getattr(other, key))
                   for key in self.__dict__)
    
    def __neq__(self, other):
        return not self.__eq__(other)

    def __repr__(self, indent="    ", use_numpy=True):
        attrs = OrderedDict((_key, fill_rep(repr(self.__dict__[_key]),
                                            _key, indent=" "*8))
                            for _key in sorted(self.__dict__))
        try:
            # Simplify a common use-case of equally spaced points
            dxs = np.diff(self.coords)
            if use_numpy and np.allclose(dxs.mean(), dxs):
                attrs['coords'] = "np.linspace({}, {}, {})".format(
                    self.coords[0], self.coords[-1], len(self.coords))
        except:
            pass
        
        return (("{}(\n" + indent).format(self.__class__.__name__)
                + (",\n" + indent).join(
                    ["{}={}".format(attr, attrs[attr]) for attr in attrs])
                + ")")


class Coords(Dimension):
    """

    Examples
    ========
    >>> Coords(coords=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ...        unit=['MeV/fm**3'])
    Coords(
        coords=np.linspace(0.1, 0.6, 6),
        unit=['MeV/fm**3'])
    """
    def __init__(self, coords, unit='1'):
        self.unit = unit
        assert unit == eval(repr(unit))
        Dimension.__init__(self, coords=coords)


class Params(Dimension):
    """
    Examples
    ========
    >>> Params(coords=['n_0', 'e_0'],
    ...        units=['1/fm**3', 'MeV/fm**3'])
    Params(
        coords=['n_0', 'e_0'],
        units=['1/fm**3', 'MeV/fm**3'])
    """
    def __init__(self, coords, units=None, values=None):
        if units is None:
            units = ['1']*len(coords)
        if not len(units) == len(coords):
            raise ValueError(
                "coords and units must have same length " +
                "got coords={}, units={}".format(coords, units))
        if values and not len(values) == len(coords):
            raise ValueError(
                "coords and values must have same length " +
                "got coords={}, values={}".format(coords, values))
        assert units == eval(repr(units))
        self.units = units
        if values is not None:
            self.values = values
        Dimension.__init__(self, coords=coords)


class Dataset(object):
    """Dataset Management.

    Arguments
    ---------
    name : str
       Name of the dataset.  This must be a valid python name and directory
       name.  It will become an importable python package.
    path : str
       Location of dataset.
    use_numpy : bool
       If True, then the generated `__init__.py` file may use numpy to simplify
       some arrays.  (In particular, regularly spaced data will be expressed
       with `np.linspace()`).
    info : str
       Will become the docstring for the dataset.
    dvcs_verbose : bool
       If True, then the full status output of the repo will be included at the
       bottom the module.
    """
    _dim_prefix = "_dim"        # Prefix for unnamed dimensions
    
    def __init__(self, name, path=".", use_numpy=True,
                 info=None, dvcs_verbose=False):
        self._name = name
        self._path = path
        self._dims = OrderedDict()
        self._arrays = OrderedDict()
        self._use_numpy = use_numpy
        if info is None:
            info = "Generated Dataset <no info>"
        print(info)
        self._info = info
        self._dvcs_verbose = dvcs_verbose

    @property
    def _mod_dir(self):
        return os.path.join(self._path, self._name)

    @property
    def _sentinel_file(self):
        return os.path.join(self._mod_dir, _SENTINEL)

    @property
    def _classes_file(self):
        return os.path.join(self._mod_dir, '_classes.py')
        
    def add(self, info="", **kw):
        for name in kw:
            assert not (self._use_numpy and name == 'np')
            value = kw[name]
            if isinstance(value, Dimension):
                assert name.startswith(self._dim_prefix) or not name.startswith('_')
                assert name not in self._dims
                self._dims[name] = value
            elif isinstance(value, Array):
                assert not name.startswith('_')
                assert name not in self._arrays
                value.filename = "{}.npy".format(name)
                self._arrays[name] = value
            else:
                raise ValueError(
                    "{} must be of type Dimension or Array, got {}".format(
                        name, value.__class__))

    def _check_dims(self):
        """Make sure that all dims specified in arrays are present in _dims."""
        for array_name in self._arrays:
            array = self._arrays[array_name]
            for ndim, dim in enumerate(array.dims):
                if isinstance(dim, str):
                    if dim not in self._dims:
                        raise ValueError("Dim {} in array {} not found."
                                         .format(dim, array_name))
                else:
                    dim_name = None
                    for _dim_name in self._dims:
                        if dim == self._dims[_dim_name]:
                            dim_name = _dim_name
                    if not dim_name:
                        # No matching dimension found.
                        n = 0
                        while True:
                            dim_name = self._dim_prefix + str(n)
                            if dim_name not in self._dims:
                                break
                            n += 1
                        self.add(**{dim_name: dim})
                    logging.debug("Replace {}.dims[{}] = {}".format(
                        array_name, ndim, dim_name))
                    array.dims[ndim] = dim_name
        return
        
    def __enter__(self):
        self._write()
        return self
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._write()

    def _write(self):
        # Make sure _mod_dir  exists, and that it contains the sentinel
        if os.path.exists(self._mod_dir):
            if not os.path.exists(self._sentinel_file):
                raise ValueError(
                    ("Directory %s exists and is not a DataSet repository. " +
                     "Please choose a unique location. ") % (self._mod_dir,))
        else:
            logging.info("Making directory %s for output." % (self._mod_dir,))
            os.makedirs(self._mod_dir)
            with open(self._sentinel_file, 'w'):
                pass

        with open(self._classes_file, 'w') as f:
            f.write(_classes_py)

        init_file = os.path.join(self._mod_dir, '__init__.py')
        with open(init_file, 'w') as f:
            f.write(self._get_init_file())
            
        self._save_arrays()

    def _save_arrays(self):
        for name in self._arrays:
            array = self._arrays[name]
            np.save(os.path.join(self._mod_dir, array.filename),
                    array.data,
                    allow_pickle=False)
            
    def _get_init_file(self):
        """Return the contents of the __init__.py file.
        
        Examples
        ========
        >>> ds = Dataset('test')
        >>> ds.add(neutron_star_observable=Params(
        ...    coords=['M', 'R', 'k_2'], units=['M0', 'km', '1']))
        >>> ds.add(Pc=Coords(
        ...    coords=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ...    unit=['MeV/fm**3']))
        >>> ds.add(EoS_param=Params(
        ...    coords=['n_0', 'e_0'],
        ...    units=['1/fm**3', 'MeV/fm**3']))
        >>> ds.add(stars=Array(None,
        ...    dims=['neutron_star_observable', 'Pc'],
        ...    unit=(1, 0)))
        >>> ds.add(derivatives=Array(None,
        ...    dims=['neutron_star_observable', 'Pc', 'EoS_param'],
        ...    unit=(1, 0, -1)))
        >>> print(ds._get_init_file())
        import numpy as np
        <BLANKLINE>
        from ._classes import _Array, _Coords, _Params
        <BLANKLINE>
        <BLANKLINE>
        _dims = dict(
            neutron_star_observable=_Params(
                coords=['M', 'R', 'k_2'],
                units=['M0', 'km', '1']),
            Pc=_Coords(
                coords=np.linspace(0.1, 0.6, 6),
                unit=['MeV/fm**3']),
            EoS_param=_Params(
                coords=['n_0', 'e_0'],
                units=['1/fm**3', 'MeV/fm**3']))
        <BLANKLINE>
        <BLANKLINE>
        stars = _Array(
            'stars.npy',
            dims=['neutron_star_observable', 'Pc'],
            unit=(1, 0),
            _dims=_dims)
        <BLANKLINE>
        <BLANKLINE>
        derivatives = _Array(
            'derivatives.npy',
            dims=['neutron_star_observable', 'Pc', 'EoS_param'],
            unit=(1, 0, -1),
            _dims=_dims)
        <BLANKLINE>
        <BLANKLINE>
        del _Array, _Coords, _Params, _dims, np
        """
        self._check_dims()

        dvcs_id, dvcs_status = self.get_version_info()
        docstring = _DOCSTRING.format(info=self._info,
                                      now=str(datetime.datetime.now()))
        if dvcs_id:
            docstring += dvcs_id

        docstring = multiline_string_repr(docstring)
            
        lines = [
            "from ._classes import _Array, _Coords, _Params\n\n",
            "_dims = dict(",
            "{DIMS})",
            "{ARRAYS}\n\n",
            "del _Array, _Coords, _Params, _dims"
        ]

        DIMS = ",\n".join(
            ["    {}=_{}".format(
                _name,
                self._dims[_name].__repr__(indent="        ",
                                           use_numpy=self._use_numpy))
             for _name in self._dims])

        ARRAYS = [
            "\n\n{} = _{}".format(
                _name,
                self._arrays[_name].__repr__(indent="    ",
                                             use_numpy=self._use_numpy))
            for _name in self._arrays]
        
        # Add "_dims=_dims" to each arrat
        for n, array in enumerate(ARRAYS):
            assert array[-1] == ")"
            ARRAYS[n] = array[:-1] + ",\n    _dims=_dims)"
        ARRAYS = "\n".join(ARRAYS)
        
        _init_file = "\n".join(lines).format(DIMS=DIMS, ARRAYS=ARRAYS)
        if self._use_numpy and "np." in _init_file:
            _init_file = "import numpy as np\n\n" + _init_file + ", np"
        _init_file = docstring + _init_file + "\n"
        if self._dvcs_verbose:
            _init_file = "\n".join([_init_file, multiline_string_repr(dvcs_status)])
        
        return _init_file

    def get_version_info(self):
        """Return a (hg_id, hg_status) strings with information about the
        repository versions if they exist.
        """
        ids = []
        status = []
        try:
            hg_id = subprocess.check_output(['hg', 'id']).strip()
            ids.append("Mercural Id (hg id): {}".format(hg_id))
            if self._dvcs_verbose:
                hg_status = subprocess.check_output(['hg', 'status']).strip()
                status.append("$ hg status")
                status.append(hg_status)
        except subprocess.CalledProcessError:
            pass

        try:
            git_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
            ids.append("Git Id (git rev-parse HEAD): {}".format(git_id))
            if self._dvcs_verbose:
                git_status = subprocess.check_output(
                    ['git', 'status', '--porcelain']).strip()
                status.append("$ git status --procelain")
                status.append(git_status)
        except subprocess.CalledProcessError:
            git_id = None

        return "\n".join(ids), "\n".join(status)
        

_classes_py = '''
from collections import OrderedDict
import os.path
import textwrap


DATA_DIR = os.path.dirname(__file__)


def fill_rep(repr, name="", indent=" "*8):
    indent = indent + " "*len(name + "=[")
    wrapper = textwrap.TextWrapper(
        width=80,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False)
    return wrapper.fill(repr)[len(indent):]


class _Array(object):
    """Represents an array stored on disk.

    Attributes
    ----------
    filename : str
       Location of data file.
    dims : [str]
       List of dimension names describing each dimension of the array.  The
       actual _Dimension objects are obtained from the `_dims` dictionary in
       the model.
    coords : None, {key: arraylike}
       Optional dict of coordinate values associated with each dimension.  If
       not provided, then the coords will be obtained from the _Dimension in
       the module dictionary.  This can be used if the specified array is
       tabulated at a different set of abscissa.
    _dims : dict(dim_name, Dimension):
       Dictionary mapping the names in `dims` to actual dimension objects.
    """
    def __init__(self, filename, dims, _dims, unit='1', coords=None):
        self.filename = filename
        self.unit = unit
        self.dims = [_dims[_dim] for _dim in dims]
        self._data = None

    @property
    def data(self):
        if self._data is None:
            import numpy as np
            self._data = np.load(os.path.join(DATA_DIR, self.filename))
        return self._data

    def __repr__(self, indent="    ", use_numpy=False):
        return (("{}(\\n" + indent).format(self.__class__.__name__)
                + (",\\n" + indent).join(
                    [repr(self.filename)]
                    +
                    ["{}={}".format(attr, fill_rep(repr(getattr(self, attr)),
                                                   attr,
                                                   indent=" "*4))
                     for attr in sorted(self.__dict__)
                     if attr not in set(['_data', 'filename'])])
                + ")")


class _Dimension(object):
    def __init__(self, coords):
        self.coords = coords

    def __repr__(self, indent="    ", use_numpy=True):
        attrs = OrderedDict((_key, fill_rep(repr(self.__dict__[_key]),
                                            _key, indent=" "*8))
                            for _key in sorted(self.__dict__))
        try:
            # Simplify a common use-case of equally spaced points
            import numpy as np
            dxs = np.diff(self.coords)
            if use_numpy and np.allclose(dxs.mean(), dxs):
                attrs['coords'] = "np.linspace({}, {}, {})".format(
                    self.coords[0], self.coords[-1], len(self.coords))
        except:
            pass
        
        return (("{}(\\n" + indent).format(self.__class__.__name__)
                + (",\\n" + indent).join(
                    ["{}={}".format(attr, attrs[attr]) for attr in attrs])
                + ")")


class _Coords(_Dimension):
    def __init__(self, coords, unit='1'):
        self.unit = unit
        _Dimension.__init__(self, coords=coords)


class _Params(_Dimension):
    def __init__(self, coords, units=None, values=None):
        if units is None:
            units = ['1']*len(coords)
        self.units = units
        self.values = values
        _Dimension.__init__(self, coords=coords)

    def __repr__(self, split=False):
        res = []
        for n, coord in enumerate(self.coords):
            if self.values and self.units:
                unit = self.units[n]
                if unit == '1':
                    res.append("{}={}".format(coord, self.values[n]))
                else:
                    res.append("{}={}*{}".format(coord, self.values[n], unit))
            elif self.values:
                res.append("{}={}".format(coord, self.values[n]))
            elif self.units:
                unit = self.units[n]
                if unit == '1':
                    res.append("{}".format(coord, self.units[n]))
                else:
                    res.append("{} [{}]".format(coord, self.units[n]))
            else:
                res.append("{}".format(coord))
        if split:
            return "\\n".join(res)
        else:
            return "Param({})".format(res)

    def __str__(self):
        return self.__repr__(split=True)
'''
