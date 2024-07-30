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

#if defined(ARDUINO) && defined(ARDUINO_ARCH_ESP32)
#   include <bootloader_random.h>
#endif

#if (defined(__GNUC__) || \
    defined(__GNUG__) || \
    defined(__clang__) || \
    defined(_MSC_VER)) && \
    !defined(ARDUINO)
#   include <cstring>
#endif

#include <diwa.h>
#include <diwa_conv.h>

#if (defined(__GNUC__) || \
    defined(__GNUG__) || \
    defined(__clang__) || \
    defined(_MSC_VER)) && \
    !defined(ARDUINO)

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
    this->activation = DiwaActivationFunc::sigmoid;
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

    #if defined(ARDUINO) && defined(ARDUINO_ARCH_ESP32) 
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

    this->inputNeurons = inputNeurons;
    this->hiddenLayers = hiddenLayers;
    this->hiddenNeurons = hiddenNeurons;
    this->outputNeurons = outputNeurons;

    this->weightCount = hiddenWeightCount + outputWeightCount;
    this->neuronCount = inputNeurons + hiddenNeurons * hiddenLayers + outputNeurons;

    if(randomizeWeights) {
        DiwaError error;
        if((error = this->initializeWeights()) != NO_ERROR)
            return error;
    }

    this->outputs = this->weights + this->weightCount;
    this->deltas = this->outputs + this->neuronCount;

    if(randomizeWeights)
        this->randomizeWeights();

    return NO_ERROR;
}

DiwaError Diwa::initializeWeights() {
    #if defined(ARDUINO) && defined(ARDUINO_ARCH_ESP32)
    if(psramFound())
        this->weights = (double*) ps_malloc(sizeof(double) * (this->weightCount * this->neuronCount));
    else this->weights = (double*) malloc(sizeof(double) * (this->weightCount * this->neuronCount));
    #else
    this->weights = (double*) malloc(sizeof(double) * (this->weightCount * this->neuronCount));
    #endif

    if(this->weights == NULL)
        return MALLOC_FAILED;

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
            *outputs++ = this->activation(sum);
        }

        return returnValues;
    }

    for(int j = 0; j < this->hiddenNeurons; ++j) {
        double sum = *weights++ * -1.0;

        for(int k = 0; k < this->inputNeurons; ++k)
            sum += *weights++ * inputs[k];
        *outputs++ = this->activation(sum);
    }

    inputs += this->inputNeurons;
    for(int h = 1; h < this->hiddenLayers; ++h) {
        for(int j = 0; j < this->hiddenNeurons; ++j) {
            double sum = *weights++ * -1.0;
            
            for(int k = 0; k < this->hiddenNeurons; ++k)
                sum += *weights++ * inputs[k];
            *outputs++ = this->activation(sum);
        }

        inputs += this->hiddenNeurons;
    }

    double* returnValue = outputs;
    for(int j = 0; j < this->outputNeurons; ++j) {
        double sum = *weights++ * -1.0;

        for(int k = 0; k < this->hiddenNeurons; ++k)
            sum += *weights++ * inputs[k];
        *outputs++ = this->activation(sum);;
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
    uint8_t magic[4];
    annFile.read(magic, 4);

    if(memcmp(magic, "diwa", 4) == 1)
        return INVALID_MAGIC_NUMBER;

    uint8_t temp_int[4];

    annFile.read(temp_int, 4);
    this->inputNeurons = DiwaConv::u8aToInt(temp_int);

    annFile.read(temp_int, 4);
    this->hiddenNeurons = DiwaConv::u8aToInt(temp_int);

    annFile.read(temp_int, 4);
    this->hiddenLayers = DiwaConv::u8aToInt(temp_int);

    annFile.read(temp_int, 4);
    this->outputNeurons = DiwaConv::u8aToInt(temp_int);

    annFile.read(temp_int, 4);
    this->weightCount = DiwaConv::u8aToInt(temp_int);

    annFile.read(temp_int, 4);
    this->neuronCount = DiwaConv::u8aToInt(temp_int);

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

        if((error = this->initializeWeights()) != NO_ERROR)
            return error;
    }

    uint8_t temp_db[8];
    for(int i = 0; i < this->weightCount; i++) {
        annFile.read(temp_db, 8);
        this->weights[i] = DiwaConv::u8aToDouble(temp_db);
    }

    return NO_ERROR;
}

DiwaError Diwa::saveToFile(File annFile) {
    const uint8_t* magic_signature = new uint8_t[4] {'d', 'i', 'w', 'a'};
    annFile.write(magic_signature, 4);

    annFile.write(DiwaConv::intToU8a(this->inputNeurons), 4);
    annFile.write(DiwaConv::intToU8a(this->hiddenNeurons), 4);
    annFile.write(DiwaConv::intToU8a(this->hiddenLayers), 4);
    annFile.write(DiwaConv::intToU8a(this->outputNeurons), 4);

    annFile.write(DiwaConv::intToU8a(this->weightCount), 4);
    annFile.write(DiwaConv::intToU8a(this->neuronCount), 4);

    for(int i = 0; i < this->weightCount; i++)
        annFile.write(DiwaConv::doubleToU8a(this->weights[i]), 8);

    annFile.flush();
    return NO_ERROR;
}

#elif defined(__GNUC__) || \
    defined(__GNUG__) || \
    defined(__clang__) || \
    defined(_MSC_VER)

DiwaError Diwa::loadFromFile(std::ifstream& annFile) {
    if(!annFile.is_open())
        return STREAM_NOT_OPEN;

    uint8_t magic[5];
    annFile.read(reinterpret_cast<char*>(magic), 4);

    if(std::string(reinterpret_cast<char*>(magic), 4) != "diwa")
        return INVALID_MAGIC_NUMBER;

    uint8_t temp_int[5];

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->inputNeurons = DiwaConv::u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->hiddenNeurons = DiwaConv::u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->hiddenLayers = DiwaConv::u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->outputNeurons = DiwaConv::u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->weightCount = DiwaConv::u8aToInt(temp_int);

    annFile.read(reinterpret_cast<char*>(temp_int), 4);
    this->neuronCount = DiwaConv::u8aToInt(temp_int);

    {
        DiwaError error;
        if((error = this->initialize(
            this->inputNeurons,
            this->hiddenLayers,
            this->hiddenNeurons,
            this->outputNeurons,
            true
        )) != NO_ERROR)
            return error;

        if((error = this->initializeWeights()) != NO_ERROR)
            return error;
    }

    uint8_t temp_db[9];
    for(int i = 0; i < this->weightCount; i++) {
        annFile.read(reinterpret_cast<char*>(temp_db), 8);
        this->weights[i] = DiwaConv::u8aToDouble(temp_db);
    }

    return NO_ERROR;
}

DiwaError Diwa::saveToFile(std::ofstream& annFile) {
    if(!annFile.is_open())
        return STREAM_NOT_OPEN;

    const uint8_t* magic_signature = new uint8_t[4] {'d', 'i', 'w', 'a'};
    writeToStream(annFile, magic_signature, 4);

    writeToStream(annFile, DiwaConv::intToU8a(this->inputNeurons), 4);
    writeToStream(annFile, DiwaConv::intToU8a(this->hiddenNeurons), 4);
    writeToStream(annFile, DiwaConv::intToU8a(this->hiddenLayers), 4);
    writeToStream(annFile, DiwaConv::intToU8a(this->outputNeurons), 4);

    writeToStream(annFile, DiwaConv::intToU8a(this->weightCount), 4);
    writeToStream(annFile, DiwaConv::intToU8a(this->neuronCount), 4);

    for(int i = 0; i < this->weightCount; i++)
        writeToStream(annFile, DiwaConv::doubleToU8a(this->weights[i]), 8);
    return NO_ERROR;
}

#endif

inline bool Diwa::testInference(double *testInput, double *testExpectedOutput) {
    double* testInference = this->inference(testInput);
    bool correctOutput = true;

    for(int j = 0; j < this->outputNeurons; j++)
        if(testInference[j] < 0.5 && testExpectedOutput[j] != 0)
            correctOutput = false;

    return correctOutput;
}

double Diwa::calculateAccuracy(double *testInput, double *testExpectedOutput, int epoch) {
    int correctInference = 0;
    for(int i = 0; i < epoch; i++)
        if(this->testInference(testInput, testExpectedOutput))
            correctInference++;

    return (double) correctInference / epoch;
}

double Diwa::calculateLoss(double *testInput, double *testExpectedOutput, int epoch) {
    return 1.0 - this->calculateAccuracy(testInput, testExpectedOutput, epoch);
}

void Diwa::setActivationFunction(diwa_activation activation) {
    this->activation = activation;
}

diwa_activation Diwa::getActivationFunction() const {
    return this->activation;
}

int Diwa::recommendedHiddenNeuronCount() {
    if(this->inputNeurons <= 0 || this->outputNeurons <= 0)
        return -1;

    return sqrt(this->inputNeurons * this->outputNeurons);
}

int Diwa::recommendedHiddenLayerCount(int numSamples, int alpha) {
    if(this->inputNeurons <= 0 ||
        this->outputNeurons <= 0 ||
        numSamples <= 0 || alpha <= 0)
        return -1;

    int count = numSamples / (alpha * (this->inputNeurons + this->outputNeurons));
    if(count < 1)
        return -1;

    return count;
}