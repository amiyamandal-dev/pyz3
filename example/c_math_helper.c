// Example C implementation for demonstrating C/C++ integration with pyz3

#include "c_math_helper.h"

int c_add(int a, int b) {
    return a + b;
}

int c_multiply(int a, int b) {
    return a * b;
}

double c_divide(double a, double b) {
    if (b == 0.0) {
        return 0.0;  // Simple error handling
    }
    return a / b;
}

int c_factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * c_factorial(n - 1);
}

int c_fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return c_fibonacci(n - 1) + c_fibonacci(n - 2);
}
