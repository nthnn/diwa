/*
 * This file is part of the Diwa library.
 * Copyright (c) 2024 Nathanne Isip
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/** 
 * @file 
 * @author [Nathanne Isip](https://github.com/nthnn)
 * @brief Utility functions for data conversion in the Diwa neural network library.
 *
 * This file provides utility functions for converting data between different formats,
 * such as integers and doubles to byte arrays, and vice versa. These functions are
 * used to facilitate data serialization and deserialization in the Diwa library.
 */

#ifndef DIWA_UTIL_H
#define DIWA_UTIL_H

/** @cond */
typedef union {
    double d;
    uint8_t b[8];
} double_p;

/**
 * @brief Utility class for data conversion operations.
 *
 * The DiwaConv class provides static methods for converting data between different
 * formats, such as integers to byte arrays, doubles to byte arrays, and vice versa.
 * These conversion methods are essential for data serialization and deserialization
 * within the Diwa library, enabling seamless integration of neural network functionality
 * with various data formats.
 */
class DiwaConv final {
public:
    /**
     * @brief Convert integer value to byte array.
     *
     * This method converts an integer value to a byte array.
     *
     * @param value The integer value to be converted.
     * @return Pointer to the byte array representing the integer value.
     */
    static inline uint8_t* intToU8a(int value) {
        uint8_t* bytes = new uint8_t[4];

        bytes[0] = (value >> 0) & 0xFF;
        bytes[1] = (value >> 8) & 0xFF;
        bytes[2] = (value >> 16) & 0xFF;
        bytes[3] = (value >> 24) & 0xFF;

        return bytes;
    }

    /**
     * @brief Convert byte array to integer value.
     *
     * This method converts a byte array to an integer value.
     *
     * @param bytes The byte array to be converted.
     * @return The integer value represented by the byte array.
     */
    static inline int u8aToInt(uint8_t bytes[4]) {
        int result = 0;

        result |= bytes[0];
        result |= bytes[1] << 8;
        result |= bytes[2] << 16;
        result |= bytes[3] << 24;

        return result;
    }

    /**
     * @brief Convert double value to byte array.
     *
     * This method converts a double value to a byte array.
     *
     * @param value The double value to be converted.
     * @return Pointer to the byte array representing the double value.
     */
    static inline uint8_t* doubleToU8a(double value) {
        double_p db;
        db.d = value;

        uint8_t* bytes = new uint8_t[8];
        for(uint8_t i = 0; i < 8; ++i)
            bytes[i] = db.b[i];

        return bytes;
    }

    /**
     * @brief Convert byte array to double value.
     *
     * This method converts a byte array to a double value.
     *
     * @param bytes The byte array to be converted.
     * @return The double value represented by the byte array.
     */
    static inline double u8aToDouble(uint8_t bytes[8]) {
        double_p db;
        for(uint8_t i = 0; i < 8; ++i)
            db.b[i] = bytes[i];

        return db.d;
    }
};

#endif  // DIWA_CONV_H