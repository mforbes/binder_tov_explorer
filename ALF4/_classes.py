
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
        return (("{}(\n" + indent).format(self.__class__.__name__)
                + (",\n" + indent).join(
                    [repr(getattr(self, 'filename', self._data))]
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
        
        return (("{}(\n" + indent).format(self.__class__.__name__)
                + (",\n" + indent).join(
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
        if values is not None:
            if hasattr(values, 'tolist'):
                values = values.tolist()
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
            return "\n".join(res)
        else:
            return "Param({})".format(res)

    def __str__(self):
        return self.__repr__(split=True)
