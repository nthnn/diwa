#ifndef DIWA_UTIL_H
#define DIWA_UTIL_H

/** @cond */
typedef union {
    double d;
    uint8_t b[8];
} double_p;

class DiwaConv final {
public:
    static inline uint8_t* intToU8a(int value) {
        uint8_t* bytes = new uint8_t[4];

        bytes[0] = (value >> 0) & 0xFF;
        bytes[1] = (value >> 8) & 0xFF;
        bytes[2] = (value >> 16) & 0xFF;
        bytes[3] = (value >> 24) & 0xFF;

        return bytes;
    }

    static inline int u8aToInt(uint8_t bytes[4]) {
        int result = 0;

        result |= bytes[0];
        result |= bytes[1] << 8;
        result |= bytes[2] << 16;
        result |= bytes[3] << 24;

        return result;
    }

    static inline uint8_t* doubleToU8a(double value) {
        double_p db;
        db.d = value;

        uint8_t* bytes = new uint8_t[8];
        for(uint8_t i = 0; i < 8; ++i)
            bytes[i] = db.b[i];

        return bytes;
    }

    static inline double u8aToDouble(uint8_t bytes[8]) {
        double_p db;
        for(uint8_t i = 0; i < 8; ++i)
            db.b[i] = bytes[i];

        return db.d;
    }
};

#endif