import time
from concurrent.futures import ThreadPoolExecutor

from example import gil


# --8<-- [start:gil]
def test_gil():
    now = time.time()
    with ThreadPoolExecutor(10) as pool:
        for _ in range(10):
            # Sleep for 100ms
            pool.submit(gil.sleep, 100)

    # This should take ~10 * 100ms. Add some leniency and check for >900ms.
    duration = time.time() - now
    assert duration > 0.9


def test_gil_release():
    now = time.time()
    with ThreadPoolExecutor(10) as pool:
        for _ in range(10):
            pool.submit(gil.sleep_release, 100)

    # This should take ~1 * 100ms. Add some leniency and check for <500ms.
    duration = time.time() - now
    assert duration < 0.5


# --8<-- [end:gil]
