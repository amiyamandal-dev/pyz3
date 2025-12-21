from example import buffers


def test_view():
    buffer = buffers.ConstantBuffer(1, 10)
    view = memoryview(buffer)
    for i in range(10):
        assert view[i] == 1
    view.release()


# --8<-- [start:sum]
def test_sum():
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert buffers.sum(arr) == 15


# --8<-- [end:sum]
