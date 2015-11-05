import collections
# algorithm by:
# http://people.cs.uct.ac.za/~ksmith/articles/sliding_window_minimum.html#sliding-window-minimum-algorithm


def sliding_window_minimum(array, k=5):
    window = collections.deque()
    for i, array_element in enumerate(array):
        while window and window[-1][0] >= array_element:
            window.pop()
        window.append((array_element, i))

        while window[0][1] <= i - k:
            window.popleft()

        yield window[0][0]
