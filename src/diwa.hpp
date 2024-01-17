/*
 * This file is part of the SIM900 Arduino Shield library.
 * Copyright (c) 2023 Nathanne Isip
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
    #include <Arduino.h>
#elif defined(__GNUC__) || !defined(__clang__) || defined(__clang__)
    #include <stdlib.h>
    #include <cstring>
#endif

#include <assert.h>
#include <math.h>

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

    float *weights;     /**< Array to store weights */
    float *outputs;     /**< Array to store neuron outputs */
    float *deltas;      /**< Array to store delta values during training */

    /**
     * 
     * @brief Randomize the weights in the network.
     * 
     */
    void randomizeWeights();

public:
    /**
     * 
     * @brief Constructor for Diwa Artificial Neural Network class.
     *
     * @param inputNeurons Number of input neurons.
     * @param hiddenLayers Number of hidden layers.
     * @param hiddenNeurons Number of neurons in each hidden layer.
     * @param outputNeurons Number of output neurons.
     * 
     */
    Diwa(
        int inputNeurons,
        int hiddenLayers,
        int hiddenNeurons,
        int outputNeurons
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
    float* inference(float *inputs);

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
        float learningRate,
        float *inputNeurons,
        float *outputNeurons
    );
};

#endif  // DIWA_H