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

#ifdef ARDUINO
#   include <bootloader_random.h>
#else
#   include <cstring>
#endif

#include <diwa.h>

typedef union {
    double d;
    uint8_t b[8];
} double_p;

static inline double diwaSigmoid(double value) {
    if(value < -45.0) return 0;
    if(value > 45.0) return 1;

    return 1.0 / (1.0 + exp(-value));
}

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

#ifndef ARDUINO

static inline void writeToStream(std::ofstream& stream, const uint8_t* data, size_t size) {
    for(size_t i = 0; i < size; i++)
        stream.write(
            reinterpret_cast<const char*>(&data[i]),
            sizeof(uint8_t)
        );

    delete[] data;
}

#endif

Diwa::Diwa() {
    this->initialize(0, 0, 0, 0);
}

Diwa::~Diwa() {
    free(this->weights);
}

inline void Diwa::randomizeWeights() {
    for(int i = 0; i < this->weightCount; i++)
        #ifdef ARDUINO
        this->weights[i] = (((double) random()) / RAND_MAX) - 0.5;
        #else
        this->weights[i] = (((double) rand()) / RAND_MAX) - 0.5;
        #endif
}

DiwaError Diwa::initialize(
    int inputNeurons,
    int hiddenLayers,
    int hiddenNeurons,
    int outputNeurons,
    bool randomizeWeights
) {
    if(inputNeurons == 0 &&
        hiddenLayers == 0 &&
        hiddenNeurons == 0 &&
        outputNeurons == 0)
        return NO_ERROR;

    #ifdef ARDUINO
    if(ESP.getFreePsram() != 0)
        return NO_ESP_PSRAM;
    #endif

    #ifdef ARDUINO
    bootloader_random_enable();
    randomSeed(esp_random());
    bootloader_random_disable();
    #endif

    const int hiddenWeightCount = hiddenLayers ?
        (inputNeurons + 1) * hiddenNeurons +
            (hiddenLayers - 1) * (hiddenNeurons + 1) *
            hiddenNeurons : 0;
    const int outputWeightCount = (hiddenLayers ?
        (hiddenNeurons + 1) : (inputNeurons + 1)
    ) * outputNeurons;

    const int weightCount = hiddenWeightCount + outputWeightCount;
    const int neuronCount = inputNeurons + hiddenNeurons * hiddenLayers + outputNeurons;

    this->inputNeurons = inputNeurons;
    this->hiddenLayers = hiddenLayers;
    this->hiddenNeurons = hiddenNeurons;
    this->outputNeurons = outputNeurons;

    this->weightCount = weightCount;
    this->neuronCount = neuronCount;

    #ifdef ARDUINO
    this->weights = (double*) ps_malloc(sizeof(double) * (weightCount * neuronCount));
    #else
    this->weights = (double*) malloc(sizeof(double) * (weightCount * neuronCount));
    #endif

    this->outputs = this->weights + this->weightCount;
    this->deltas = this->outputs + this->neuronCount;

    if(randomizeWeights)
        this->randomizeWeights();

    return NO_ERROR;
}

double* Diwa::inference(double *inputNeurons) {
    double *weights = this->weights;
    double *inputs = this->outputs;
    double *outputs = this->outputs + this->inputNeurons;

    memcpy(this->outputs, inputNeurons, sizeof(double) * this->inputNeurons);
    if(!this->hiddenLayers) {
        double *returnValues = outputs;
        for(int j = 0; j < this->outputNeurons; ++j) {
            double sum = *weights++ * -1.0;

            for(int k = 0; k < this->inputNeurons; ++k)
                sum += *weights++ * inputs[k];
            *outputs++ = diwaSigmoid(sum);
        }

        return returnValues;
    }

    for(int j = 0; j < this->hiddenNeurons; ++j) {
        double sum = *weights++ * -1.0;

        for(int k = 0; k < this->inputNeurons; ++k)
            sum += *weights++ * inputs[k];
        *outputs++ = diwaSigmoid(sum);
    }

    inputs += this->inputNeurons;
    for(int h = 1; h < this->hiddenLayers; ++h) {
        for(int j = 0; j < this->hiddenNeurons; ++j) {
            double sum = *weights++ * -1.0;
            
            for(int k = 0; k < this->hiddenNeurons; ++k)
                sum += *weights++ * inputs[k];
            *outputs++ = diwaSigmoid(sum);
        }

        inputs += this->hiddenNeurons;
    }

    double* returnValue = outputs;
    for(int j = 0; j < this->outputNeurons; ++j) {
        double sum = *weights++ * -1.0;

        for(int k = 0; k < this->hiddenNeurons; ++k)
            sum += *weights++ * inputs[k];
        *outputs++ = diwaSigmoid(sum);;
    }

    return returnValue;
}

void Diwa::train(double learningRate, double *inputNeurons, double *outputNeurons) {
    this->inference(inputNeurons);

    {
        double *outputs =
            this->outputs +
            this->inputNeurons +
            this->hiddenNeurons *
            this->hiddenLayers;
        double *deltas =
            this->deltas +
            this->hiddenNeurons *
            this->hiddenLayers;
        double *training = outputNeurons;

        for(int j = 0; j < this->outputNeurons; ++j) {
            *deltas++ = (*training - *outputs) *
                *outputs * (1.0 - *outputs);
            outputs++, training++;
        }
    }

    for(int h = this->hiddenLayers - 1; h >= 0; --h) {
        double *outputs =
            this->outputs +
            this->inputNeurons +
            (h * this->hiddenNeurons);

        double *deltas =
            this->deltas +
            (h * this->hiddenNeurons);

        double *firstDelta =
            this->deltas +
            ((h + 1) * this->hiddenNeurons);

        double *firstWeight =
            this->weights +
            ((this->inputNeurons + 1) * this->hiddenNeurons) +
            ((this->hiddenNeurons + 1) * this->hiddenNeurons * h);

        for(int j = 0; j < this->hiddenNeurons; ++j) {
            double delta = 0;

            for(int k = 0;
                k < (h == this->hiddenLayers - 1 ?
                    this->outputNeurons :
                    this->hiddenNeurons);
                ++k) {
                int weightIndex = k * (this->hiddenNeurons + 1) + (j + 1);

                double forwardDelta = firstDelta[k];
                double forwardWeight = firstWeight[weightIndex];

                delta += forwardDelta * forwardWeight;
            }

            *deltas = *outputs * (1.0 - *outputs) * delta;
            deltas++, outputs++;
        }
    }

    {
        double *deltas =
            this->deltas +
            this->hiddenNeurons *
            this->hiddenLayers;

        double *weights =
            this->weights +
            (this->hiddenLayers ?
                (this->inputNeurons + 1) * this->hiddenNeurons +
                (this->hiddenNeurons + 1) * this->hiddenNeurons *
                (this->hiddenLayers - 1) : 0);

        double *firstOutput =
            this->outputs +
            (this->hiddenLayers ?
                (this->inputNeurons + this->hiddenNeurons *
                    (this->hiddenLayers - 1)) : 0);

        for(int j = 0; j < this->outputNeurons; ++j) {
            *weights++ += *deltas * learningRate * -1.0;

            for(int k = 1;
                k < (this->hiddenLayers ?
                    this->hiddenNeurons : this->inputNeurons) + 1;
                ++k)
                *weights++ += *deltas * learningRate * firstOutput[k - 1];

            deltas++;
        }
    }

    for(int h = this->hiddenLayers - 1; h >= 0; --h) {
        double *deltas = this->deltas +
            (h * this->hiddenNeurons);

        double *firstInput = this->outputs +
            (h ? this->inputNeurons +
                this->hiddenNeurons * (h - 1) : 0);

        double *weights =
            this->weights + (h ?
                (this->inputNeurons + 1) * this->hiddenNeurons +
                (this->hiddenNeurons + 1) * this->hiddenNeurons *
                (h - 1) : 0);

        for(int j = 0; j < this->hiddenNeurons; ++j) {
            *weights += *deltas * learningRate * -1.0;

            for(int k = 1;
                k < (h == 0 ?
                    this->inputNeurons : this->hiddenNeurons) + 1;
                ++k)
                *weights++ += *deltas * learningRate * firstInput[k - 1];

            deltas++;
        }
    }
}

#ifdef ARDUINO

DiwaError Diwa::loadFromFile(File annFile) {
    return NO_ERROR;
}

DiwaError Diwa::saveToFile(File annFile) {
    return NO_ERROR;
}

#else

DiwaError Diwa::loadFromFile(std::ifstream& annFile) {
    if(!annFile.is_open())
        return STREAM_NOT_OPEN;

    uint8_t magic[5];
    annFile.read(reinterpret_cast<char*>(magic), 4);

    if(std::string(reinterpret_cast<char*>(magic), 4) != "diwa")
        return INVALID_MAGIC_NUMBER;

    uint8_t temp_int[5];

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->inputNeurons = u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->hiddenNeurons = u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->hiddenLayers = u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->outputNeurons = u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->weightCount = u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->neuronCount = u8aToInt(temp_int);

    {
        DiwaError error;
        if((error = this->initialize(
                this->inputNeurons,
                this->hiddenLayers,
                this->hiddenNeurons,
                this->outputNeurons,
                false
            )) != NO_ERROR)
            return error;
    }

    uint8_t temp_db[9];
    for(int i = 0; i < this->weightCount; i++) {
        annFile.read(reinterpret_cast<char*>(temp_db), 8);
        this->weights[i] = u8aToDouble(temp_db);
    }

    return NO_ERROR;
}

DiwaError Diwa::saveToFile(std::ofstream& annFile) {
    if(!annFile.is_open())
        return STREAM_NOT_OPEN;

    const uint8_t* magic_signature = new uint8_t[4] {'d', 'i', 'w', 'a'};
    writeToStream(annFile, magic_signature, 4);

    writeToStream(annFile, intToU8a(this->inputNeurons), 4);
    writeToStream(annFile, intToU8a(this->hiddenNeurons), 4);
    writeToStream(annFile, intToU8a(this->hiddenLayers), 4);
    writeToStream(annFile, intToU8a(this->outputNeurons), 4);

    writeToStream(annFile, intToU8a(this->weightCount), 4);
    writeToStream(annFile, intToU8a(this->neuronCount), 4);

    for(int i = 0; i < this->weightCount; i++)
        writeToStream(annFile, doubleToU8a(this->weights[i]), 8);

    return NO_ERROR;
}

#endif