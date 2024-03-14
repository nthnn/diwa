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

#ifndef DIWA_H
#define DIWA_H

#ifdef ARDUINO
#   include <Arduino.h>
#   include <FS.h>
#else
#   include <fstream>
#   include <stdint.h>
#endif

#include <math.h>

typedef enum {
    NO_ERROR,
    INVALID_PARAM_VALUES,
    MODEL_READ_ERROR,
    MODEL_SAVE_ERROR,
    INVALID_MAGIC_NUMBER,
    STREAM_NOT_OPEN,

    #ifdef ARDUINO
    NO_ESP_PSRAM
    #endif
} DiwaError;

/**
 * 
 * @class Diwa
 * @brief Lightweight Feedforward Artificial Neural Network
 *        (ANN) library tailored for microcontrollers.
 *
 * The Diwa library is designed to provide a simple yet
 * effective implementation of a Feedforward Artificial
 * Neural Network (ANN) for resource-constrained
 * microcontroller environments such as ESP8266, ESP32,
 * and similar development boards.
 *
 * @note This library is primarily intended for lightweight
 *       applications. For more intricate tasks, consider
 *       using advanced machine learning libraries on more
 *       powerful platforms.
 * 
 */
class Diwa {
private:
    int inputNeurons;   /**< Number of input neurons */
    int hiddenNeurons;  /**< Number of neurons in each hidden layer */
    int hiddenLayers;   /**< Number of hidden layers */
    int outputNeurons;  /**< Number of output neurons */

    int weightCount;    /**< Total number of weights in the network */
    int neuronCount;    /**< Total number of neurons in the network */

    double *weights;     /**< Array to store weights */
    double *outputs;     /**< Array to store neuron outputs */
    double *deltas;      /**< Array to store delta values during training */

    /**
     * 
     * @brief Randomize the weights in the network.
     * 
     */
    void randomizeWeights();

public:
    Diwa();
    ~Diwa();

    DiwaError initialize(
        int inputNeurons,
        int hiddenLayers,
        int hiddenNeurons,
        int outputNeurons,
        bool randomizeWeights = true
    );

    /**
     * 
     * @brief Perform inference on the neural network.
     *
     * Given an array of input values, this method computes
     * and returns an array of output values through the
     * neural network.
     *
     * @param inputs Array of input values for the neural network.
     * @return Array of output values after inference.
     * 
     */
    double* inference(double *inputs);

    /**
     * 
     * @brief Train the neural network using backpropagation.
     *
     * This method facilitates the training of the neural
     * network by adjusting its weights based on the provided
     * input and target output values.
     *
     * @param learningRate Learning rate for the training process.
     * @param inputNeurons Array of input values for training.
     * @param outputNeurons Array of target output values for training.
     * 
     */
    void train(
        double learningRate,
        double *inputNeurons,
        double *outputNeurons
    );

    #ifdef ARDUINO
    DiwaError loadFromFile(File annFile);
    #else
    DiwaError loadFromFile(std::ifstream& annFile);
    #endif

    #ifdef ARDUINO
    DiwaError saveToFile(File annFile);
    #else
    DiwaError saveToFile(std::ofstream& annFile);
    #endif
};

#endif  // DIWA_H