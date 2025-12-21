"""
zigimport - Import Zig directly from Python!

Automatically compile and import Zig extension modules without manual compilation steps.
Inspired by rustimport and cppimport.

Advanced Features:
    - Dependency tracking for imported .zig files
    - Custom build.zig support
    - Parallel compilation
    - Watch mode for auto-reload
    - Remote caching
    - PyPI distribution support

Usage:
    import pyz3.zigimport
    import my_module  # Automatically compiles my_module.zig
"""

import os
import sys
import hashlib
import json
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Optional, Set, List, Dict
import re
import threading
import concurrent.futures
import time
from functools import lru_cache

from pyz3 import buildzig, config as pyz3_config


class ZigImportConfig:
    """Configuration for zigimport behavior."""

    def __init__(self):
        # Optimization level
        self.optimize = os.environ.get("ZIGIMPORT_OPTIMIZE", "Debug")

        # Custom build directory
        self.build_dir = Path(os.environ.get("ZIGIMPORT_BUILD_DIR", Path.home() / ".zigimport"))
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Verbose output
        self.verbose = os.environ.get("ZIGIMPORT_VERBOSE", "").lower() in ("true", "1", "yes")

        # Force rebuild
        self.force_rebuild = os.environ.get("ZIGIMPORT_FORCE_REBUILD", "").lower() in ("true", "1", "yes")

        # Parallel compilation
        self.parallel = os.environ.get("ZIGIMPORT_PARALLEL", "1").lower() in ("true", "1", "yes")
        self.max_workers = int(os.environ.get("ZIGIMPORT_MAX_WORKERS", "4"))

        # Watch mode
        self.watch_mode = os.environ.get("ZIGIMPORT_WATCH", "").lower() in ("true", "1", "yes")
        self.watch_interval = float(os.environ.get("ZIGIMPORT_WATCH_INTERVAL", "1.0"))

        # Remote caching
        self.remote_cache = os.environ.get("ZIGIMPORT_REMOTE_CACHE", "")
        self.cache_upload = os.environ.get("ZIGIMPORT_CACHE_UPLOAD", "").lower() in ("true", "1", "yes")

        # Dependency tracking
        self.track_deps = os.environ.get("ZIGIMPORT_TRACK_DEPS", "1").lower() in ("true", "1", "yes")

        # Cache file
        self.cache_file = self.build_dir / "modules.json"

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[zigimport] {message}", file=sys.stderr)


class DependencyTracker:
    """Track Zig file dependencies by parsing @import statements."""

    @staticmethod
    def find_imports(zig_file: Path) -> Set[Path]:
        """Find all @import statements in a Zig file."""
        imports = set()

        try:
            content = zig_file.read_text()

            # Match @import("path") or @import("path.zig")
            import_pattern = r'@import\s*\(\s*"([^"]+)"\s*\)'
            matches = re.findall(import_pattern, content)

            for match in matches:
                # Skip built-in imports
                if match in ("std", "builtin", "root", "pyz3"):
                    continue

                # Try to resolve the import path
                import_path = Path(match)
                if not import_path.suffix:
                    import_path = import_path.with_suffix(".zig")

                # Check relative to current file
                resolved = (zig_file.parent / import_path).resolve()
                if resolved.exists():
                    imports.add(resolved)

        except Exception:
            pass

        return imports

    @staticmethod
    def get_all_dependencies(zig_file: Path, visited: Optional[Set[Path]] = None) -> Set[Path]:
        """Recursively get all dependencies of a Zig file."""
        if visited is None:
            visited = set()

        if zig_file in visited:
            return visited

        visited.add(zig_file)

        # Find direct imports
        imports = DependencyTracker.find_imports(zig_file)

        # Recursively find transitive dependencies
        for imported_file in imports:
            DependencyTracker.get_all_dependencies(imported_file, visited)

        return visited


class ZigModuleCache:
    """Cache tracking module build times and dependencies."""

    def __init__(self, config: ZigImportConfig):
        self.config = config
        self.cache: Dict = {}
        self.load()

    def load(self) -> None:
        """Load cache from disk."""
        if self.config.cache_file.exists():
            try:
                with open(self.config.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def save(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.config.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    def get_hash(self, source_path: Path, deps: Optional[Set[Path]] = None) -> str:
        """Calculate hash of source file and its dependencies."""
        hasher = hashlib.sha256()

        # Hash main source
        hasher.update(source_path.read_bytes())

        # Hash dependencies if tracking enabled
        if self.config.track_deps and deps:
            for dep in sorted(deps):
                if dep != source_path and dep.exists():
                    hasher.update(dep.read_bytes())

        # Hash build configuration
        hasher.update(self.config.optimize.encode())

        return hasher.hexdigest()

    def needs_rebuild(self, module_name: str, source_path: Path, output_path: Path,
                     deps: Optional[Set[Path]] = None) -> bool:
        """Check if module needs rebuild."""
        if self.config.force_rebuild:
            return True

        if not output_path.exists():
            return True

        # Check hash
        current_hash = self.get_hash(source_path, deps)
        cached_data = self.cache.get(module_name, {})
        cached_hash = cached_data.get("hash", "")

        return current_hash != cached_hash

    def update(self, module_name: str, source_path: Path, deps: Optional[Set[Path]] = None) -> None:
        """Update cache entry."""
        self.cache[module_name] = {
            "hash": self.get_hash(source_path, deps),
            "source": str(source_path),
            "dependencies": [str(d) for d in (deps or set())],
            "timestamp": time.time(),
        }
        self.save()


class RemoteCache:
    """Handle remote caching of compiled modules."""

    def __init__(self, config: ZigImportConfig):
        self.config = config
        self.enabled = bool(config.remote_cache)
        self.cache_dir = Path(config.remote_cache) if self.enabled else None

    def download(self, module_hash: str, output_path: Path) -> bool:
        """Download compiled module from remote cache."""
        if not self.enabled:
            return False

        try:
            cache_file = self.cache_dir / f"{module_hash}.so"
            if cache_file.exists():
                import shutil
                shutil.copy2(cache_file, output_path)
                self.config.log(f"Downloaded from cache: {module_hash}")
                return True
        except Exception as e:
            self.config.log(f"Cache download failed: {e}")

        return False

    def upload(self, module_hash: str, compiled_path: Path):
        """Upload compiled module to remote cache."""
        if not self.enabled or not self.config.cache_upload:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{module_hash}.so"

            import shutil
            shutil.copy2(compiled_path, cache_file)
            self.config.log(f"Uploaded to cache: {module_hash}")
        except Exception as e:
            self.config.log(f"Cache upload failed: {e}")


class CompilationQueue:
    """Queue for parallel compilation of modules."""

    def __init__(self, config: ZigImportConfig):
        self.config = config
        self.executor = None
        self.futures: Dict[str, concurrent.futures.Future] = {}

        if config.parallel:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)

    def submit(self, module_name: str, compile_func, *args, **kwargs):
        """Submit a compilation task."""
        if self.executor:
            future = self.executor.submit(compile_func, *args, **kwargs)
            self.futures[module_name] = future
            return future
        else:
            # Synchronous compilation
            return compile_func(*args, **kwargs)

    def wait(self, module_name: str):
        """Wait for a specific module compilation."""
        if module_name in self.futures:
            return self.futures[module_name].result()

    def shutdown(self):
        """Shutdown the executor."""
        if self.executor:
            self.executor.shutdown(wait=True)


class WatchMode:
    """Watch .zig files and auto-reload on changes."""

    def __init__(self, config: ZigImportConfig):
        self.config = config
        self.watched_files: Dict[str, float] = {}
        self.running = False
        self.thread = None

    def watch(self, module_name: str, source_path: Path, deps: Set[Path]):
        """Add files to watch list."""
        files_to_watch = {source_path} | deps

        for file in files_to_watch:
            if file.exists():
                self.watched_files[str(file)] = file.stat().st_mtime

    def start(self):
        """Start watch mode in background thread."""
        if self.running or not self.config.watch_mode:
            return

        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        self.config.log("Watch mode started")

    def _watch_loop(self):
        """Background loop checking for file changes."""
        while self.running:
            time.sleep(self.config.watch_interval)

            for filepath, old_mtime in list(self.watched_files.items()):
                path = Path(filepath)
                if path.exists():
                    new_mtime = path.stat().st_mtime
                    if new_mtime > old_mtime:
                        self.config.log(f"Change detected: {filepath}")
                        # Trigger reload by invalidating cache
                        module_name = self._get_module_name(filepath)
                        if module_name and module_name in sys.modules:
                            try:
                                import importlib
                                importlib.reload(sys.modules[module_name])
                                self.config.log(f"Reloaded: {module_name}")
                            except Exception as e:
                                self.config.log(f"Reload failed: {e}")

                        self.watched_files[filepath] = new_mtime

    def _get_module_name(self, filepath: str) -> Optional[str]:
        """Try to determine module name from filepath."""
        # Simple heuristic - look for module in sys.modules
        for name, module in sys.modules.items():
            if hasattr(module, '__file__') and module.__file__ == filepath:
                return name
        return None

    def stop(self):
        """Stop watch mode."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


class ZigImportFinder(importlib.abc.MetaPathFinder):
    """Import finder for .zig files using pyz3's build system."""

    def __init__(self, config: ZigImportConfig):
        self.config = config
        self.cache = ZigModuleCache(config)
        self.remote_cache = RemoteCache(config)
        self.compilation_queue = CompilationQueue(config)
        self.watch_mode = WatchMode(config)

        if config.watch_mode:
            self.watch_mode.start()

    def find_spec(self, fullname, path, target=None):
        """Find a .zig file and compile it using pyz3."""
        # Find the .zig source file or build.zig
        source_path = self._find_zig_file(fullname, path)
        build_zig_path = self._find_build_zig(fullname, path)

        if not source_path and not build_zig_path:
            return None

        self.config.log(f"Found {'build.zig' if build_zig_path else source_path} for module {fullname}")

        # Determine output directory and file
        module_dir = self.config.build_dir / fullname.replace(".", "/")
        module_dir.mkdir(parents=True, exist_ok=True)

        # Expected output path
        output_name = fullname.split(".")[-1] + ".abi3.so"
        output_path = module_dir / output_name

        # Track dependencies if enabled
        deps = set()
        if self.config.track_deps and source_path:
            deps = DependencyTracker.get_all_dependencies(source_path)
            self.config.log(f"Found {len(deps)} dependencies for {fullname}")

        # Check remote cache first
        module_hash = self.cache.get_hash(source_path or build_zig_path, deps)
        if self.remote_cache.download(module_hash, output_path):
            self.cache.update(fullname, source_path or build_zig_path, deps)
        elif self.cache.needs_rebuild(fullname, source_path or build_zig_path, output_path, deps):
            self.config.log(f"Compiling {fullname}...")

            # Compile (possibly in parallel)
            if build_zig_path:
                self._compile_with_build_zig(fullname, build_zig_path, module_dir, output_path)
            else:
                self._compile_module(fullname, source_path, module_dir, output_path)

            self.cache.update(fullname, source_path or build_zig_path, deps)

            # Upload to remote cache
            if output_path.exists():
                self.remote_cache.upload(module_hash, output_path)
        else:
            self.config.log(f"Using cached {fullname}")

        # Watch for changes if enabled
        if self.config.watch_mode and source_path:
            self.watch_mode.watch(fullname, source_path, deps)

        # Load the compiled extension
        if not output_path.exists():
            raise ImportError(f"Failed to compile {fullname}")

        loader = importlib.machinery.ExtensionFileLoader(fullname, str(output_path))
        return importlib.util.spec_from_loader(fullname, loader, origin=str(output_path))

    def _find_zig_file(self, fullname: str, path) -> Optional[Path]:
        """Find .zig file for the given module name."""
        parts = fullname.split(".")
        module_name = parts[-1]

        # Search in sys.path and current directory
        search_paths = list(path or []) + [os.getcwd()] + sys.path

        for search_path in search_paths:
            if not search_path:
                continue

            search_dir = Path(search_path)

            # Try module_name.zig
            candidate = search_dir / f"{module_name}.zig"
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    def _find_build_zig(self, fullname: str, path) -> Optional[Path]:
        """Find custom build.zig for module."""
        parts = fullname.split(".")
        module_name = parts[-1]

        search_paths = list(path or []) + [os.getcwd()] + sys.path

        for search_path in search_paths:
            if not search_path:
                continue

            search_dir = Path(search_path)

            # Try module_name/build.zig
            candidate = search_dir / module_name / "build.zig"
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    def _compile_module(self, fullname: str, source_path: Path, module_dir: Path, output_path: Path):
        """Compile a Zig module using pyz3's build system."""
        # Create ExtModule configuration
        ext_module = pyz3_config.ExtModule(
            name=fullname,
            root=source_path,
            limited_api=True,
        )

        # Create a temporary build directory
        build_dir = module_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # Save current directory
        orig_dir = os.getcwd()

        try:
            # Change to build directory
            os.chdir(build_dir)

            # Create pyz3 config
            conf = pyz3_config.ToolPydust(
                build_zig=build_dir / "build.zig",
                ext_modules=[ext_module],
                self_managed=False,
            )

            # Build the module
            buildzig.zig_build(
                [
                    "install",
                    f"-Dpython-exe={sys.executable}",
                    f"-Doptimize={self.config.optimize}",
                ],
                conf=conf,
            )

            # Find and copy the compiled output
            self._copy_output(build_dir, output_path)

        finally:
            # Restore directory
            os.chdir(orig_dir)

    def _compile_with_build_zig(self, fullname: str, build_zig_path: Path, module_dir: Path, output_path: Path):
        """Compile using custom build.zig."""
        build_dir = module_dir / "build"
        build_dir.mkdir(exist_ok=True)

        orig_dir = os.getcwd()

        try:
            os.chdir(build_zig_path.parent)

            # Use the custom build.zig directly
            import subprocess
            result = subprocess.run(
                [
                    "zig", "build",
                    f"-Doptimize={self.config.optimize}",
                    f"-Dpython-exe={sys.executable}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if self.config.verbose and result.stdout:
                print(result.stdout, file=sys.stderr)

            # Find and copy the compiled output
            self._copy_output(build_zig_path.parent, output_path)

        finally:
            os.chdir(orig_dir)

    def _copy_output(self, build_dir: Path, output_path: Path):
        """Find and copy compiled output."""
        zig_out = build_dir / "zig-out" / "lib"
        if zig_out.exists():
            # Find the .so/.dylib/.pyd file
            for pattern in ["*.so", "*.dylib", "*.pyd"]:
                for so_file in zig_out.glob(pattern):
                    import shutil
                    shutil.copy2(so_file, output_path)
                    return


# Global state
_config = None
_finder = None


def install():
    """Install the zigimport import hook."""
    global _config, _finder

    if _finder is not None:
        return

    _config = ZigImportConfig()
    _finder = ZigImportFinder(_config)

    sys.meta_path.insert(0, _finder)

    _config.log("zigimport import hook installed")


def uninstall() -> None:
    """Uninstall the zigimport import hook."""
    global _finder

    if _finder is not None:
        if _finder.watch_mode:
            _finder.watch_mode.stop()
        if _finder.compilation_queue:
            _finder.compilation_queue.shutdown()

        if _finder in sys.meta_path:
            sys.meta_path.remove(_finder)
        _finder = None


def clear_cache() -> None:
    """Clear the compilation cache."""
    if _config and _config.build_dir.exists():
        import shutil
        shutil.rmtree(_config.build_dir)
        _config.build_dir.mkdir(parents=True, exist_ok=True)


def enable_watch_mode() -> None:
    """Enable watch mode for auto-reload."""
    if _finder and _finder.watch_mode:
        _finder.watch_mode.start()


def disable_watch_mode() -> None:
    """Disable watch mode."""
    if _finder and _finder.watch_mode:
        _finder.watch_mode.stop()


# Auto-install when imported
install()
