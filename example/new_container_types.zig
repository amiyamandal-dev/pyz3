// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const std = @import("std");
const py = @import("pyz3");

const root = @This();

/// Demonstrates PyDefaultDict
pub fn defaultdict_example() !py.PyObject {
    const builtins = try py.import(root, "builtins");
    defer builtins.decref();
    const int_type = try builtins.get("int");
    defer int_type.decref();

    const dd = try py.PyDefaultDict(root).new(int_type);
    const dd_dict = dd.asDict();

    // Access a missing key, which should call the int factory and return 0
    const missing_val = try dd_dict.getItem(py.PyObject, "missing");
    if (missing_val) |val| val.decref();

    // Set a value in the dict
    const five = try py.PyLong.create(5);
    defer five.obj.decref();
    try dd_dict.setItem("count", five.obj);

    return dd.obj;
}

/// Demonstrates PyCounter
pub fn counter_example() !py.PyObject {
    const iterable = try py.PyString.create("abracadabra");
    defer iterable.obj.decref();

    const counter = try py.PyCounter(root).fromIterable(iterable.obj);
    
    // Get the 2 most common elements
    const common = try counter.mostCommon(2);
    return common.obj;
}

/// Demonstrates PyDeque
pub fn deque_example() !py.PyObject {
    const d = try py.PyDeque(root).new();

    const one = try py.PyLong.create(1);
    defer one.obj.decref();
    const two = try py.PyLong.create(2);
    defer two.obj.decref();

    try d.append(one.obj);
    try d.appendLeft(two.obj);
    try d.rotate(1);

    return d.obj;
}

/// Demonstrates PyFraction
pub fn fraction_example() !f64 {
    const f1 = try py.PyFraction(root).new(1, 2);
    defer f1.obj.decref();
    const f2 = try py.PyFraction(root).new(3, 4);
    defer f2.obj.decref();

    const sum = try f1.add(f2);
    defer sum.obj.decref();

    return sum.asFloat();
}

/// Demonstrates PyEnum
pub fn enum_example() !py.PyObject {
    const Color = try py.PyEnum(root).new("Color", &.{ "RED", "GREEN", "BLUE" });
    defer Color.obj.decref();

    const RED = try Color.getMember("RED");
    defer RED.decref();

    const name = try py.PyEnum(root).getMemberName(RED);
    const name_str = try py.PyString.create(name);
    return name_str.obj;
}

// Demonstrates PyAsyncGenerator check
// Disabled - PyAsyncGenerator has compilation issues
// pub fn async_generator_check(args: struct { obj: py.PyObject }) !bool {
//     return py.PyAsyncGenerator(root).check(args.obj);
// }

comptime {
    py.rootmodule(root);
}
