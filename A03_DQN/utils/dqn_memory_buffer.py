# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 02:22:57 2021

@author: Luiz Resende Silva
"""
from typing import Sequence, Any, Optional, Union
import numpy as np
import collections
import sys


class MemoryBuffer(object):
    """
    Class object MemoryBuffer creates a modified deque object.

    The object can have a given fixed length ```max_size``` or not and when instantiating, the object
    can be empty or initialized.

    Parameters
    ----------
    data : Union[Sequence, None], optional
        A sequence or iterable to be converted to object type MemoryBuffer. The default is None.
    max_size : Union[int, None], optional
        Maximum length of object. The default is None.
    seed : Union[int, None], optional
        The seed for the random number generator responsible for generating samples. The default is 895359.

    Arguments
    ---------
    __data : Sequence
    __seed : int
    __rng : numpy.random._generator.Generator
    __max_size : int
    __memory_container : collections.deque
    size : int
    nbytes : int

    Methods
    -------
    __getitem__(index)
        Returns the element in the given index, making object indexable
    __len__()
        Returns length of object
    __iter__()
        Makes object iterable
    __str__()
        Returns object type name
    __repr__()
        Returns printable object type and current elements
    seed(seed)
        Returns the class object seed or sets a new seed to the class object
    add(new_element)
        Adds new_element to object
    pop_first()
        Removes leftmost element from object
    pop_last()
        Removes rightmost element from object
    remove_first(item)
        Removes first appearance of item from object
    remove_all(item)
        Removes all appearances of item from object
    random_samples(sample_size)
        Returns list of size sample_size with randomly sampled elements from object
    tolist()
        Returns deque object converted to list
    tonumpy()
        Returns deque object converted to numpy.ndarray
    """

    def __init__(self, data: Optional[Any] = None, max_size: Optional[Union[int, None]] = None,
                 seed: Optional[int] = 895359) -> None:
        self.__seed = seed
        self.__rng = np.random.default_rng(self.__seed)
        self.__max_size = max_size
        if (self.__max_size is not None):  # Setting maxlen automatically removed oldest values
            if (data is not None):
                self.__memory_container = collections.deque(data, maxlen=self.__max_size)
            else:
                self.__memory_container = collections.deque(maxlen=self.__max_size)
        else:
            if (data is not None):
                self.__memory_container = collections.deque(data)
            else:
                self.__memory_container = collections.deque()
        self.size = len(self.__memory_container)
        self.nbytes = sys.getsizeof(self.__memory_container)

    def __getitem__(self, index: int) -> Any:
        """
        Private method to make object indexable.

        Parameters
        ----------
        index : int or list or numpy.ndarray
            An int for single element, an array of int for multiple elements or an array of bool of length self.size for
            multiple elements.

        Raises
        ------
        IndexError
            'IndexError: index must be integer, iterable or slice. Got %s instead...'.
            'IndexError: index must be integer, iterable or slice. Got %s instead...'

        Returns
        -------
        Sequence
            List with indexed elements.
        """
        if (isinstance(index, int)):
            return self.__memory_container.__getitem__(index)
        elif (isinstance(index, list) or isinstance(index, np.ndarray)):
            if (isinstance(index[0], bool) and (len(index) == self.size)):
                items = np.arange(0, self.size, dtype=np.int32)[index]
                return [self.__memory_container[i] for i in items]
            elif ((len(index) <= self.size) and
                  (isinstance(index[0], int) or (isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.integer)))):
                return [self.__memory_container[i] for i in index]
            else:
                i_type = str(type(index[0]))
                raise IndexError('IndexError: index must be iterable of Union[int, bool]. Got %s instead...' % i_type)
        elif (isinstance(index, slice)):
            result = self.tolist().__getitem__(index)
            return list(result)
        else:
            raise IndexError('IndexError: index must be integer, iterable or slice. Got %s instead...' % str(type(index)))

    def __len__(self) -> int:
        """
        Private method to return length of container.

        Returns
        -------
        int
            Length/size of deque.
        """
        return self.size

    def __iter__(self) -> Any:
        """
        Private method to make object iterable.

        Yields
        ------
        Any
            Each element of the deque.
        """
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        """
        Private method for object name print.

        Returns
        -------
        str
            Object type name.
        """
        return str(self.__memory_container)[6:-1].replace(',', '')

    def __repr__(self) -> str:
        """
        Private method to print object type and content.

        Returns
        -------
        str
            Object type and current content.
        """
        if (self.size <= 100):
            return ('FixedMemory(%s)' % str(self.__memory_container)[6:-1])
        else:
            lst_1 = '['
            lst_2 = ''
            for i in range(0, 5, 1):
                lst_1 += (str(self.__memory_container[i]) + ', ')
                lst_2 += (', ' + str(self.__memory_container[(i - 5)]))
            st = (lst_1 + '...' + lst_2 + ']')
            return ('FixedMemory(%s)' % st)

    def seed(self, seed: Optional[Union[int, None]] = None) -> Union[int, None]:
        """
        Method to return current seed value or set new seed value for random number generator.

        Parameters
        ----------
        seed : int or None, optional
            If None, returns current value. If int, sets self.__seed to the new seed value. The default is None.

        Raises
        ------
        TypeError
            TypeError: seed must be of type int of None. Got %s instead.

        Returns
        -------
        int or None
            If seed=None, returns self.__seed. If seed is int, returns None.
        """
        if (seed is None):
            return self.__seed
        else:
            if (isinstance(seed, int)):
                self.__seed = seed
                self.__rng = np.random.default_rng(self.__seed)
                return None
            else:
                raise TypeError('TypeError: seed must be of type int of None. Got %s instead...' % str(type(seed)))

    def add(self, new_element: Any) -> None:
        """
        Method adds new element to FixedMemory object.

        Parameters
        ----------
        new_element : variable
            Can be of any type.

        Returns
        -------
        None.

        Note
        ----
        If object maximum length has been reached, the leftmost element is deleted automatically.
        """
        self.__memory_container.append(new_element)
        if (self.__max_size is None):
            self.size += 1
        elif (self.size < self.__max_size):
            self.size += 1
        self.nbytes = sys.getsizeof(self.__memory_container)

    def pop_first(self) -> Any:
        """
        Method removes the leftmost element in the object and returns it.

        Returns
        -------
        pop_elem : variables
            The element which previously occupied the leftmost position in the object.
        """
        if (self.size >= 1):
            pop_elem = self.__memory_container.popleft()
            self.size -= 1
            self.nbytes = sys.getsizeof(self.__memory_container)
            return pop_elem
        else:
            print('pop from an empty deque')
            return None

    def pop_last(self) -> Any:
        """
        Method removes the rightmost element in the object and returns it.

        Returns
        -------
        pop_elem : variables
            The element which previously occupied the rightmost position in the object.
        """
        if (self.size >= 1):
            pop_elem = self.__memory_container.pop()
            self.size -= 1
            self.nbytes = sys.getsizeof(self.__memory_container)
            return pop_elem
        else:
            print('pop from an empty deque')
            return None

    def remove_first(self, item: int) -> None:
        """
        Method removes the first appearence of the item from the object.

        Parameters
        ----------
        item : Any
            The element whose first occurence is to be removed.

        Returns
        -------
        None.
        """
        if (item in self.__memory_container):
            self.__memory_container.remove(item)
            self.size -= 1
            self.nbytes = sys.getsizeof(self.__memory_container)
        else:
            print(str(item) + ' not present in the object')

    def remove_all(self) -> None:
        """
        Method removes all all items from the object.

        Parameters
        ----------
        None

        Returns
        -------
        None.
        """
        self.__memory_container.clear()
        self.size = len(self.__memory_container)
        self.nbytes = sys.getsizeof(self.__memory_container)

    def random_samples(self, sample_size: int) -> Sequence:
        """
        Method randomly selects and returns a list containing a given number of elements.

        Parameters
        ----------
        sample_size : int
            The number of elements to be randomly selected from the object.

        Raises
        ------
        ValueError
            ValueError: sample_size larger than number of elements in object. Choose sample_size <= %d

        Returns
        -------
        memory_batch : Sequence
            Sequence containing the randomly sampled elements.
        """
        if (self.size >= sample_size):
            indices = self.__rng.choice(self.size, size=sample_size, replace=False, p=None).astype(dtype=int).tolist()
            memory_batch = self.__getitem__(indices)
            return memory_batch
        else:
            raise ValueError(('ValueError: sample_size larger than available elements in object. ')
                             + ('Choose sample_size <= %d' % self.size))

    def tolist(self) -> Sequence:
        """
        Method returns object converted to python type list.

        Returns
        -------
        list
            Deque object converted to list.
        """
        return list(self.__memory_container)

    def tonumpy(self) -> np.ndarray:
        """
        Method returns object converted to python type numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            Deque object converted to numpy.ndarray.
        """
        return np.array(list(self.__memory_container))
