//
// Created by lz on 24-9-27.
//

#include "utils.h"

int count_significant_digits(double num) {
    // Handle the special case for 0
    if (num == 0.0) {
        return 1; // Zero has exactly one significant digit
    }
    int count = 0;
    int started = 0; // Marker indicating whether counting has started

    // Handle negative numbers
    if (num < 0) {
        num = -num;
    }
    
    // Handle fractional part
    while (num!=(int)num) {
        num *=10;
        count++;
    }
    return count;
}