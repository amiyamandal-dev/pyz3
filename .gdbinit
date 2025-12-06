# GDB initialization file for debugging pyz3 extensions
# Place this in your project root or home directory (~/.gdbinit)

# Enable Python support
python
import sys
print("\n🐛 GDB initialized for pyz3 debugging")
end

# Pretty printing
set print pretty on
set print object on
set print static-members on
set print vtbl on
set print demangle on
set demangle-style gnu-v3

# History
set history save on
set history size 10000
set history filename ~/.gdb_history

# Disable pagination for long output
set pagination off

# Python exception handling
catch throw

# Useful aliases
define pyz3-info
    info sharedlibrary
    info threads
end

define pyz3-break
    break $arg0
end

# Load Python pretty printers if available
# python sys.path.insert(0, '/path/to/pyz3/printers')
# python import pyz3_printers
# python pyz3_printers.register()

echo \n🐛 GDB ready for pyz3 debugging\n
echo    Use 'pyz3-break function_name' to set breakpoints\n
echo    Use 'pyz3-info' for module information\n
echo    Use 'bt' for backtrace\n\n
