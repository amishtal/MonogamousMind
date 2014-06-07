import array
import numbers
import operator

import deap
import deap.base

import h5py
import numpy as np


class WeightedSumFitness(deap.base.Fitness):
    """
    Fitness object that uses a weighted summation interpretation
    of multiple fitness values. That is, fitnesses are compared
    by comparing the summation of weighted individual components
    rather than a lexicographic-based comparison.
      Ex: Suppose we have two fitness values,

              f1 = (f11, f12, f13) with weights (w11, w12, w13)

          and

              f2 = (f21, f22, f23) with weights (w21, w22, w23).

          The weighted sum interpretation would compare the values

              F1 = w11*f11 + w12*f12 + w13*f13

          and

              F2 = w21*f21 + w22*f22 + w23*f23,

          while the lexicographic scheme would compare

              F11 = w11*f11 and F21 = w21*f21,

          then 

              F12 = w12*f12 and F22 = w22*f22,

          and finally 

              F13 = w13*f13 and F23 = w23*f23,

          returning True if all comparisons hold and False otherwise.

    Note: I'm not really sure why DEAP uses a lexicographic
          comparison scheme for fitness anyway. It makes weights
          irrelevant (unless individuals have differing weights)
          and I don't feel that it accuratly captures differences
          between two fitness values.

          Depending on which comparison operators are used and how
          they are used to determine the survival of individuals, it
          could have strong positive or negative effects on diversity.
          Weighted summation does not have this problem, but can be
          sensitive to the weight values and relative magnitudes of
          fitness components.
    """

    def wsum(self):
        """
        Combined the individual fitness components via
        weighted summation.
        """
        return reduce(operator.add, self.wvalues, 0.0)

    def __lt__(self, other):
        try:
            return self.wsum() < other.wsum()
        except AttributeError as e:
            print "Trying to compare a WeightedSumFitness object \
                   to a Fitness object that does not support \
                   weighted summation."
            return None

    def __lte__(self, other):
        try:
            return self.wsum() <= other.wsum()
        except AttributeError as e:
            print "Trying to compare a WeightedSumFitness object \
                   to a Fitness object that does not support \
                   weighted summation."
            return None

def dictToHDFS(ds, save_file):
    """
    Converts a dictionary (or list of dictionaries) into an
    HDFS object. Dictionary keys should be strings, with
    values being either numeric values or a list/array of
    numeric values. All dictionaries should have the same
    structure.
    """

    # Ensure we have a list of dictionaries.
    if type(ds) is not list:
        if isinstance(ds, dict):
            ds = [ds]
        else:
            raise TypeError('Error: dictToHDFS does not support {} as a \
                             a valid input type.'.format(type(ds)))
    else:
        # We have a list of something, make sure they are dicts.
        elem_types = set([type(d) for d in ds])
        if len(elem_types) > 1 or dict not in elem_types:
            raise TypeError('Error: dictToHDFS requires lists of dicts, \
                             but instead received a list containing the \
                             following types: {}.'.format(elem_types))

    # Class to facilitate storing paths as lists and
    # easily converting them to the appropriate string.
    class Path(list):
        def __str__(self):
            return '/'.join(map(str, self))

    n_elems = len(ds)

    # Get dataset names by concatenating nested dictionary keys.
    dset_paths = []
    dset_shapes = []
    def process_dict(d, path):
        for (key, value) in d.items():
            if isinstance(value, dict):
                # Nested dict, recurse into it.
                new_path = Path(path)
                new_path.append(key)
                process_dict(value, new_path)
            else:
                # Reached a non-dict value that will
                # become a dataset.
                shape = [n_elems]
                if isinstance(value, np.ndarray):
                    shape.extend(value.shape)
                elif isinstance(value, numbers.Number):
                    shape.append(1)
                elif isinstance(value, array.array):
                    shape.append(len(value))
                elif isinstance(value, list):
                    shape.append(len(value))
                    x = value
                    while isinstance(x[0], list):
                        shape.append(len(x[0]))
                        x = x[0]
                else:
                    raise TypeError('dictToHDFS has found an object of \
                                      type {}, which it does not know how \
                                      to handle.'.format(type(value)))

                new_path = Path(path)
                new_path.append(key)

                dset_paths.append(new_path)
                dset_shapes.append(shape)

    process_dict(ds[0], Path())

    print dset_paths
    print dset_shapes

    def get_nested_values(d, keys, shape):
        v = d
        for key in keys:
            v = v[key]

        if shape[-1] == 1:
            return [v]
        else:
            return v

    for (path, shape) in zip(dset_paths, dset_shapes):
        x = save_file.create_dataset(str(path), shape)#, data=[get_nested_values(d, path) for d in ds])
        x[...] = [get_nested_values(d, path, shape) for d in ds]

