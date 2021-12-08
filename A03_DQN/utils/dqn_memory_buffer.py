"""
DQN-Memory-Buffer

@author: [Luiz Resende Silva](https://github.com/luiz-resende)
@date: Created on Wed Oct 20, 2021

Creates a container aimed to store transition experiences for the Deep Q-Network algorithm. Such
object class has build-in functions to facilitate the storing the sampling processes required while
implementing the algorithm to train a Reinforcement Learning agent.

Implementation Notes:
---------------------
After extensive tests of time efficiency for the different operations implemented, the class
``MemoryBufferDeque`` was improved. However, given the time required for collections.deque
indexing is O(n), a second class object was created, now implementing a list instead, which
has indexing with O(1) bound. To bypass the O(n + 1) bound for deletion of first element and
addition of new element, the class implements a position tracking, such that the object is not
a stack, but rather a 'ring' list, i.e., the newest element's index is immediately before the
oldest element's index. With this implementation, adding a new element occurs at the same time
the oldest one is removed (in reality it is replaced by the new), making the operation have a time
bound of O(1). Such optimization of reading/writing time does not affect the algorithm since all
samples, for the implementation used, are taken at random, and a parameter keeps track of where
the object begins (i.e., oldest element), such as to virtually still consider object as a stack
when sampling from it. This would not be true for implementations such as 'prioratized experience replay'.

"""
from typing import Sequence, Any, Optional, Union
import collections
import numpy as np
import random
import sys


class MemoryBufferDeque(object):
    """
    Class object MemoryBuffer creates a modified deque object.

    The object can have a given fixed length ``capacity`` or not and when instantiating, the object
    can be empty or initialized. By passing a ``capacity``, the object uses the inbuild function of
    collection.deque, where the left-most element is automatically deleted.

    Parameters
    ----------
    data : Union[Sequence, None], optional
        A sequence or iterable to be converted to object type MemoryBuffer. The default is None.
    capacity : Union[int, None], optional
        Maximum length of object. The default is None.
    seed : Union[int, None], optional
        The seed for the random number generator responsible for generating samples. The default is 895359.

    Arguments
    ---------
    __data : Sequence
    __seed : int
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
    insert(new_element)
        Adds new_element to object
    pop_first()
        Removes leftmost element from object
    pop_last()
        Removes rightmost element from object
    remove_first(item)
        Removes first appearance of item from object
    remove_all(item)
        Removes all appearances of item from object
    clear()
        Removes all the elements from object
    random_samples(sample_size)
        Returns list of size sample_size with randomly sampled elements from object
    tolist()
        Returns deque object converted to list
    tonumpy()
        Returns deque object converted to numpy.ndarray
    """

    def __init__(self, data: Optional[Any] = None, capacity: Optional[Union[int, None]] = None,
                 seed: Optional[int] = 895359) -> None:
        self.__seed = seed
        random.seed(self.__seed)
        self.__max_size = capacity
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
            return ('MemoryBuffer(%s)' % str(self.__memory_container)[6:-1])
        else:
            lst_1 = '['
            lst_2 = ''
            for i in range(0, 5, 1):
                lst_1 += (str(self.__memory_container[i]) + ', ')
                lst_2 += (', ' + str(self.__memory_container[(i - 5)]))
            st = (lst_1 + '...' + lst_2 + ']')
            return ('MemoryBuffer(%s)' % st)

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
                random.seed(self.__seed)
                return None
            else:
                raise TypeError('TypeError: seed must be of type int of None. Got %s instead...' % str(type(seed)))

    def insert(self, new_element: Any) -> None:
        """
        Method adds new element to MemoryBuffer object.

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
            print(str(item) + ' not present in the object...')

    def remove_all(self, item: int) -> None:
        """
        Method removes all the appearence of an item from the object.

        Parameters
        ----------
        item : Any
            The element whose first occurence is to be removed.

        Returns
        -------
        None.
        """
        if (item in self.__memory_container):
            while (item in self.__memory_container):
                self.remove_first(item)
        else:
            print(str(item) + ' not present in the object...')

    def clear(self) -> None:
        """
        Method removes all elements from the object.

        Parameters
        ----------
        None

        Returns
        -------
        None.
        """
        self.__memory_container.clear()
        self.size = 0
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
            memory_batch = random.sample(self.__memory_container, k=sample_size)
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


class MemoryBuffer(object):
    """
    Class object MemoryBuffer creates a modified list object.

    The object can have a given fixed length ``capacity`` or not and when instantiating, the object
    can be empty or initialized. This object creates a 'ring' list, where a variable keeps track of the
    current storing position for faster insertion of new elements that override previous elements. In this
    sense, the container is not treated as a stack, but rather as a 'ring', where the newly inserted element
    will be immediately before the oldest contained element. Such implementation takes advantage of python
    list O(1) indexation time, meaning that all operations (insertion and sampling) will have a O(1) bound.
    A parameter, ```self.__list_start```, keeps track of where the object begins (i.e., oldest element), such
    as to virtually still consider object as a stack when sampling from it.

    Parameters
    ----------
    data : ``Union[Sequence, None]``, optional
        A sequence or iterable to be converted to object type MemoryBuffer. The default is None.
    capacity : ``Union[int, None]``, optional
        Maximum length of object. The default is None.
    seed : ``Union[int, None]``, optional
        The seed for the random number generator responsible for generating samples. The default is 895359.

    Arguments
    ---------
    __data : Sequence
    __seed : int
    __max_size : int
    __memory_container : Sequence
    __list_position : int
    __list_start : int
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
    insert(new_element)
        Adds new_element to object
    pop_first()
        Removes leftmost element from object
    pop_last()
        Removes rightmost element from object
    remove_first(item)
        Removes first appearance of item from object
    remove_all(item)
        Removes all appearances of item from object
    clear()
        Removes all elements from array
    random_samples(sample_size)
        Returns list of size sample_size with randomly sampled elements from object
    tonumpy()
        Returns deque object converted to numpy.ndarray
    """

    def __init__(self,
                 data: Optional[Any] = None,
                 capacity: Optional[Union[int, None]] = None,
                 seed: Optional[int] = 895359
                 ) -> None:
        self.__seed = seed
        random.seed(self.__seed)
        self.__max_size = capacity
        if (data is not None):
            self.__memory_container = list(data)[:self.__max_size]
        else:
            self.__memory_container = []
        self.size = len(self.__memory_container)
        self.nbytes = sys.getsizeof(self.__memory_container)
        if (self.size == 0):
            self.__list_position = 0
        elif (self.__max_size is None):
            self.__list_position = self.size
        else:
            self.__list_position = self.size % self.__max_size
        self.__list_start = 0

    def __getitem__(self,
                    index: int
                    ) -> Any:
        """
        Private method to make object indexable.

        Method accepts integer indices, slices and array of indices (integer and Boolean).

        Parameters
        ----------
        index : Union[int, slice, list, numpy.ndarray]
            An int for single element, an array of int for multiple elements or an array of bool of length self.size for
            multiple elements.

        Raises
        ------
        IndexError
            'IndexError: index must be iterable of Union[int, bool]. Got %s instead...'.
            'IndexError: index must be integer, iterable or slice. Got %s instead...'

        Returns
        -------
        Sequence
            List with indexed elements.
        """
        if ((not isinstance(index, int)) and (not isinstance(index, slice))):
            if (isinstance(index, list) or isinstance(index, np.ndarray)):
                if (isinstance(index[0], bool) and (len(index) == self.size)):
                    items = np.arange(0, self.size, dtype=np.int32)[index]
                    return [self.__memory_container[i] for i in items]
                elif ((len(index) <= self.size) and
                      (isinstance(index[0], int) or (isinstance(index, np.ndarray) and
                                                     np.issubdtype(index.dtype, np.integer)))):
                    return [self.__memory_container[i] for i in index]
                else:
                    i_type = str(type(index[0]))
                    raise IndexError('IndexError: index must be iterable of Union[int, bool]. Got %s instead...' % i_type)
            else:
                i_type = str(type(index))
                raise IndexError('IndexError: index must be integer, iterable or slice. Got %s instead...' % i_type)
        else:
            return self.__memory_container.__getitem__(index)

    def __len__(self
                ) -> int:
        """
        Private method to return length of container.

        Returns
        -------
        int
            Length/size of deque.
        """
        return self.size

    def __iter__(self
                 ) -> Any:
        """
        Private method to make object iterable.

        Yields
        ------
        Any
            Each element of the deque.
        """
        order = [((self.__list_start + i) % self.size) for i in range(self.size)]
        for i in order:
            yield self[i]

    def __str__(self
                ) -> str:
        """
        Private method for object name print.

        Returns
        -------
        str
            Object type name.
        """
        return str(self.__memory_container[self.__list_start:]
                   + self.__memory_container[:self.__list_start]).replace(',', '')

    def __repr__(self
                 ) -> str:
        """
        Private method to print object type and content.

        Returns
        -------
        str
            Object type and current content.
        """
        if (self.size <= 100):
            return ('MemoryBuffer(%s)' % str(self.__memory_container[self.__list_start:]
                                             + self.__memory_container[:self.__list_start]))
        else:
            lst_1 = '['
            lst_2 = ''
            for i in range(0, 5, 1):
                lst_1 += (str(self.__memory_container[(i + self.__list_start)]) + ', ')
                lst_2 += (', ' + str(self.__memory_container[(i + self.__list_start - 5)]))
            st = (lst_1 + '...' + lst_2 + ']')
            return ('MemoryBuffer(%s)' % st)

    def __array__(self,
                  dtype=None
                  ) -> np.ndarray:
        """
        Method returns object converted to python type numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            The array object converted to numpy.ndarray.
        """
        if (dtype is not None):
            out = np.array(self, dtype=dtype)
        else:
            out = np.array(self)
        return out

    def seed(self,
             seed: Optional[Union[int, None]] = None
             ) -> Union[int, None]:
        """
        Method to return current seed value or set new seed value for random number generator.

        Parameters
        ----------
        seed : ``int`` or ``None``, optional
            If ``None``, returns current value. If int, sets ``self.__seed`` to the new seed value.
            The default is ``None``.

        Raises
        ------
        TypeError
            TypeError: seed must be of type int of None. Got %s instead.

        Returns
        -------
        int or None
            If ``seed=None``, returns ``self.__seed``. If seed is ``int``, returns ``None``.
        """
        if (seed is None):
            return self.__seed
        else:
            if (isinstance(seed, int)):
                self.__seed = seed
                random.seed(self.__seed)
                return None
            else:
                raise TypeError('TypeError: seed must be of type int of None. Got %s instead...' % str(type(seed)))

    def insert(self,
               new_element: Any
               ) -> None:
        """
        Method inserts new element to MemoryBuffer object.

        Parameters
        ----------
        new_element : Any
            Any type of element.

        Returns
        -------
        None.

        Note
        ----
        If object has a maximum length set, the newly added elements will replace the older elements
        following a circular path, i.e., when the last position is reached, the next element will be
        added to position 0 and so on has been reached.
        """
        if (self.__max_size is None):
            self.__memory_container.append(new_element)
            self.__list_position = self.size
            self.size += 1
            self.nbytes = sys.getsizeof(self.__memory_container)
        else:
            if (self.size < self.__max_size):
                self.__memory_container.append(None)
                self.size += 1
            self.__memory_container[self.__list_position] = new_element
            self.__list_position = (self.__list_position + 1) % self.__max_size
            if (((self.__list_position - 1) == self.__list_start) and (self.size == self.__max_size)):
                self.__list_start = (self.__list_start + 1) % self.__max_size
            self.nbytes = sys.getsizeof(self.__memory_container)

    def pop_first(self
                  ) -> Any:
        """
        Method removes the leftmost element in the object and returns it.

        Returns
        -------
        pop_elem : Any
            The element which previously occupied the leftmost position in the object.
        """
        if (self.size > 0):
            pop_elem = self.__memory_container.pop(self.__list_start)
            self.size -= 1
            if (self.size <= self.__list_start):
                self.__list_start = 0
            self.__list_position -= 1
            if (self.__list_position < 0):
                if (self.size == 0):
                    self.__list_position = 0
                    self.__list_start = 0
                else:
                    self.__list_position = self.size
            self.nbytes = sys.getsizeof(self.__memory_container)
            return pop_elem
        else:
            print('Trying to pop() from an empty array...')
            return None

    def pop_last(self
                 ) -> Any:
        """
        Method removes the rightmost element in the object and returns it.

        Returns
        -------
        pop_elem : variables
            The element which previously occupied the rightmost position in the object.

        Warning
        -------
        This method can mess with the parameters ``self.__list_start`` and ``self.__list_position``.
        """
        if (self.size > 0):
            pop_elem = self.__memory_container.pop(self.__list_position - 1)
            self.size -= 1
            self.__list_position -= 1
            if (self.__list_position < self.__list_start):
                self.__list_start -= 1
            if (self.__list_position < 0):
                if (self.size == 0):
                    self.__list_position = 0
                    self.__list_start = 0
                else:
                    self.__list_position = self.size % self.__max_size
            self.nbytes = sys.getsizeof(self.__memory_container)
            return pop_elem
        else:
            print('Trying to pop() from an empty array...')
            return None

    def remove_first(self,
                     item: Any
                     ) -> None:
        """
        Method removes the first appearence of the item from the array.

        Parameters
        ----------
        item : Any
            The element whose first occurence is to be removed.

        Returns
        -------
        None.

        Warning
        -------
        This method can mess with the parameters ``self.__list_start`` and ``self.__list_position``.
        """
        if (item in self.__memory_container):
            first_idx = self.__memory_container.index(item)
            self.__memory_container.remove(item)
            self.size -= 1
            if (self.__max_size is not None):
                if ((self.__list_start == first_idx) and (self.__list_start == self.size)):
                    self.__list_start = 0
                elif (self.__list_start > first_idx):
                    self.__list_start -= 1
            self.__list_position -= 1
            if (self.__list_position < 0):
                if (self.size == 0):
                    self.__list_position = 0
                    self.__list_start = 0
                else:
                    self.__list_position = self.size % self.__max_size
            self.nbytes = sys.getsizeof(self.__memory_container)
        else:
            print(str(item) + ' not present in the array...')

    def remove_all(self,
                   item: Any
                   ) -> None:
        """
        Method removes all the occurrences of a given item from the array.

        Parameters
        ----------
        item : Any
            The element whose first occurence is to be removed.

        Returns
        -------
        None.

        Warning
        -------
        This method can mess with the parameters ``self.__list_start`` and ``self.__list_position``.
        """
        if (item in self.__memory_container):
            while (item in self.__memory_container):
                self.remove_first(item)
        else:
            print(str(item) + ' not present in the array...')

    def clear(self
              ) -> None:
        """
        Method removes all items from the array.

        Parameters
        ----------
        None

        Returns
        -------
        None.
        """
        self.__memory_container = []
        self.__list_position = 0
        self.__list_start = 0
        self.size = 0
        self.nbytes = sys.getsizeof(self.__memory_container)

    def random_samples(self,
                       sample_size: int,
                       look_start: Optional[bool] = False
                       ) -> Sequence:
        r"""
        Method randomly selects and returns a list containing a given number of elements.

        Parameters
        ----------
        sample_size : ``int``
            The number of elements to be randomly selected from the object.
        look_start : ``bool``, optional
            Whether or not to take into account the shifted starting position of the list. This will
            increase the method's runtime in a few milliseconds. The default is ``False``.

        Raises
        ------
        ValueError
            ValueError: sample_size larger than number of elements in object. Choose ``sample_size`` :math:`\leq` ``size``

        Returns
        -------
        samples : Sequence
            Sequence containing the randomly sampled elements.
        """
        if (self.size >= sample_size):
            if (look_start):
                indices = random.sample(list(range(0, self.size, 1)), k=sample_size)
                indices = [int((self.__list_start + idx) % self.size) for idx in indices]
                samples = self.__getitem__(indices)
            else:
                samples = random.sample(self.__memory_container, k=sample_size)
            return samples
        else:
            raise ValueError(('ValueError: sample_size larger than available elements in object. ')
                             + ('Choose sample_size <= %d' % self.size))

    def tolist(self) -> Sequence:
        """
        Method returns object converted to python type list.

        Returns
        -------
        list
            Object converted to list.
        """
        return list(self)
