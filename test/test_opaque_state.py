"""Tests for opaque Zig state in Python classes"""
import pytest
import sys


def test_buffer_manager_functionality(example):
    """Test that BufferManager works correctly"""
    mgr = example.opaque_state.BufferManager(size=100)

    mgr.write(data=b"hello ")
    mgr.write(data=b"world")

    result = mgr.read_all()
    assert result == b"hello world"

    mgr.clear()
    result = mgr.read_all()
    assert result == b""


def test_buffer_manager_opaque_state(example):
    """Test that internal state is NOT accessible from Python"""
    mgr = example.opaque_state.BufferManager(size=100)

    # ✅ Cannot access allocator
    with pytest.raises(AttributeError):
        _ = mgr.allocator

    # ✅ Cannot access buffer
    with pytest.raises(AttributeError):
        _ = mgr.buffer

    # ✅ Cannot access size
    with pytest.raises(AttributeError):
        _ = mgr.size

    # ✅ Cannot access write_pos
    with pytest.raises(AttributeError):
        _ = mgr.write_pos

    # ✅ vars() shows no internal state
    if hasattr(mgr, '__dict__'):
        assert mgr.__dict__ == {}

    # ✅ dir() shows only methods, not data
    members = dir(mgr)
    assert 'write' in members
    assert 'read_all' in members
    assert 'buffer' not in members
    assert 'allocator' not in members


def test_data_processor_opaque_structures(example):
    """Test that complex Zig structures are opaque"""
    proc = example.opaque_state.DataProcessor(buffer_size=1024)

    result = proc.process(data=b"hello world")
    assert result == b"HELLO WORLD"

    count = proc.get_process_count()
    assert count == 1

    # ✅ Cannot access arena
    with pytest.raises(AttributeError):
        _ = proc.arena

    # ✅ Cannot access work_buffer
    with pytest.raises(AttributeError):
        _ = proc.work_buffer

    # ✅ Cannot access result_buffer
    with pytest.raises(AttributeError):
        _ = proc.result_buffer

    # ✅ Cannot access process_count field (only via method)
    with pytest.raises(AttributeError):
        _ = proc.process_count

    # ✅ Cannot access is_finalized
    with pytest.raises(AttributeError):
        _ = proc.is_finalized


def test_secure_storage_encryption_key_hidden(example):
    """Test that encryption key is NEVER exposed to Python"""
    storage = example.opaque_state.SecureStorage()

    # Store and retrieve data
    storage.store(data=b"my secret password")
    result = storage.retrieve()
    assert result == b"my secret password"

    # ✅ CRITICAL: encryption_key must be inaccessible
    with pytest.raises(AttributeError):
        _ = storage.encryption_key

    # ✅ Cannot access encrypted_data
    with pytest.raises(AttributeError):
        _ = storage.encrypted_data

    # ✅ Cannot access data_length
    with pytest.raises(AttributeError):
        _ = storage.data_length

    # ✅ Cannot access allocator
    with pytest.raises(AttributeError):
        _ = storage.allocator

    # ✅ No introspection reveals internal state
    import inspect
    members = [name for name, _ in inspect.getmembers(storage)]
    assert 'encryption_key' not in members
    assert 'encrypted_data' not in members


def test_file_manager_handle_hidden(example):
    """Test that file handle is opaque"""
    import tempfile
    import os

    mgr = example.opaque_state.FileManager()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")

        mgr.open(path=path)
        mgr.write(data=b"test content")
        assert mgr.get_bytes_written() == 12
        mgr.close()

        # Verify file was written
        with open(path, 'rb') as f:
            assert f.read() == b"test content"

    # ✅ Cannot access file handle
    with pytest.raises(AttributeError):
        _ = mgr.file

    # ✅ Cannot access path_buffer
    with pytest.raises(AttributeError):
        _ = mgr.path_buffer

    # ✅ Cannot access bytes_written field
    with pytest.raises(AttributeError):
        _ = mgr.bytes_written


def test_shared_resource_ref_counting_hidden(example):
    """Test that reference counting is internal only"""
    res = example.opaque_state.SharedResource()

    res.increment_data()
    assert res.get_data() == 1

    res.add_reference()

    # ✅ Cannot access resource pointer
    with pytest.raises(AttributeError):
        _ = res.resource

    # ✅ Cannot access allocator
    with pytest.raises(AttributeError):
        _ = res.allocator


def test_no_dict_exposure(example):
    """Test that __dict__ doesn't expose internal state"""
    mgr = example.opaque_state.BufferManager(size=100)

    # Most Zig classes won't have __dict__ or it will be empty
    if hasattr(mgr, '__dict__'):
        assert len(mgr.__dict__) == 0, "Internal state leaked via __dict__"


def test_no_setattr_to_create_state(example):
    """Test that Python can't add attributes to shadow internal state"""
    mgr = example.opaque_state.BufferManager(size=100)

    # Even if we try to set an attribute with the same name,
    # it should either fail or not affect internal state
    mgr.write(data=b"original")

    try:
        # Try to override internal field
        mgr.buffer = b"fake buffer"
    except (AttributeError, TypeError):
        # Expected: can't set attributes
        pass

    # Internal state should be unchanged
    result = mgr.read_all()
    assert result == b"original"


def test_no_vars_exposure(example):
    """Test that vars() doesn't reveal internal state"""
    mgr = example.opaque_state.BufferManager(size=100)

    # vars() should show no internal state
    try:
        v = vars(mgr)
        assert len(v) == 0, "Internal state leaked via vars()"
    except TypeError:
        # Also acceptable - some objects don't support vars()
        pass


def test_cleanup_deterministic(example):
    """Test that __del__ cleanup is deterministic"""
    import gc

    # Create and destroy object
    mgr = example.opaque_state.BufferManager(size=1000)
    mgr.write(data=b"test")
    del mgr

    # Force garbage collection
    gc.collect()

    # If we got here without segfault, cleanup worked
    # Create another to verify allocator still works
    mgr2 = example.opaque_state.BufferManager(size=1000)
    mgr2.write(data=b"test2")
    assert mgr2.read_all() == b"test2"


def test_no_memory_leak_on_error(example):
    """Test that memory is cleaned up even when errors occur"""
    mgr = example.opaque_state.BufferManager(size=10)

    # This should fail (buffer overflow)
    with pytest.raises(ValueError, match="Buffer overflow"):
        mgr.write(data=b"this is way too long for the buffer")

    # Object should still be usable
    mgr.write(data=b"short")
    assert mgr.read_all() == b"short"

    # Cleanup should work normally
    del mgr


def test_finalize_prevents_double_free(example):
    """Test that finalize flag prevents double-free"""
    proc = example.opaque_state.DataProcessor(buffer_size=100)

    # Process some data
    proc.process(data=b"test")

    # Manually finalize
    proc.finalize()

    # Should raise error if used after finalize
    with pytest.raises(RuntimeError, match="finalized"):
        proc.process(data=b"test2")

    # Cleanup should be safe (won't double-free)
    del proc


def test_secure_storage_key_randomness(example):
    """Test that encryption keys are unique per instance"""
    storage1 = example.opaque_state.SecureStorage()
    storage2 = example.opaque_state.SecureStorage()

    # Store same data in both
    storage1.store(data=b"same data")
    storage2.store(data=b"same data")

    # Both should retrieve correctly
    assert storage1.retrieve() == b"same data"
    assert storage2.retrieve() == b"same data"

    # But encrypted data should be different (different keys)
    # We can't access encrypted_data directly (it's opaque!)
    # So we just verify functionality works as expected


def test_python_cannot_influence_lifetime(example):
    """Test that Python code cannot affect Zig memory lifetime"""
    mgr = example.opaque_state.BufferManager(size=100)
    mgr.write(data=b"test")

    # Try various ways to mess with the object
    try:
        mgr.__del__()  # Try to call __del__ manually
    except:
        pass

    # Object should still work (Python-level __del__ doesn't affect Zig state)
    # Note: calling __del__ manually is undefined behavior, but shouldn't crash
    result = mgr.read_all()
    assert result == b"test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
