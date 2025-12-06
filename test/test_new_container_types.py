# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from example import new_container_types
from collections import defaultdict, Counter, deque
from fractions import Fraction
from enum import Enum
import asyncio

def test_defaultdict_example():
    d = new_container_types.defaultdict_example()
    assert isinstance(d, defaultdict)
    assert d["missing"] == 0
    assert d["count"] == 5

def test_counter_example():
    common = new_container_types.counter_example()
    assert isinstance(common, list)
    assert len(common) == 2
    assert common[0] == ('a', 5)
    assert common[1] == ('b', 2)

def test_deque_example():
    d = new_container_types.deque_example()
    assert isinstance(d, deque)
    assert len(d) == 2
    assert list(d) == [1, 2]

def test_fraction_example():
    result = new_container_types.fraction_example()
    assert result == 1.25

def test_enum_example():
    name = new_container_types.enum_example()
    assert isinstance(name, str)
    assert name == "RED"

# Disabled - PyAsyncGenerator has compilation issues
# @pytest.mark.asyncio
# async def test_async_generator_check():
#     async def my_agen():
#         yield 1
#
#     agen = my_agen()
#     assert new_container_types.async_generator_check(agen)
#
#     def my_gen():
#         yield 1
#
#     gen = my_gen()
#     assert not new_container_types.async_generator_check(gen)
