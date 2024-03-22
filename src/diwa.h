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
 * @file diwa.h
 * @author [Nathanne Isip](https://github.com/nthnn)
 * @brief This file contains the declaration of the Diwa class, a lightweight
 *        Feedforward Artificial Neural Network (ANN) library tailored mainly
 *        for microcontrollers.
 *
 * The Diwa class allows users to initialize, train, and perform inference with
 * neural networks, as well as save and load trained models from files. It supports
 * both Arduino and non-Arduino environments, enabling seamless integration into
 * various projects.
 *
 * @note This library is primarily intended for lightweight applications. For more
 *       complex tasks requiring advanced machine learning capabilities, consider
 *       using more powerful platforms and libraries.
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

#include <diwa_activations.h>
#include <math.h>

/**
 * @enum DiwaError
 * @brief Enumeration representing various error codes
 *        that may occur during the operation of the Diwa
 *        library.
 */
typedef enum {
    NO_ERROR,               /**< No error */
    INVALID_PARAM_VALUES,   /**< Invalid parameter values */
    MODEL_READ_ERROR,       /**< Error reading model */
    MODEL_SAVE_ERROR,       /**< Error saving model */
    INVALID_MAGIC_NUMBER,   /**< Invalid magic number */
    STREAM_NOT_OPEN,        /**< Stream not open */
    MALLOC_FAILED,          /**< Memory allocation failed */
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
class Diwa final {
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

    diwa_activation activation; /**< Activation function to be used on inference */

    /**
     * @brief Randomizes the weights in the neural network.
     *
     * This function randomizes the weights in the neural network to initialize them 
     * with random values. It is typically used during the initialization of the 
     * neural network to ensure that the weights start with diverse values, which 
     * aids in learning and prevents convergence to local minima.
     */
    void randomizeWeights();

    /**
     * @brief Initializes memory for neural network weights.
     *
     * This function allocates memory for the weights of the neural network. It 
     * initializes the memory space required to store the weights, which is determined 
     * based on the architecture and parameters of the neural network. If memory allocation 
     * fails, it returns an error code indicating the failure, allowing the calling code 
     * to handle the error gracefully.
     *
     * @return DiwaError The corresponding DiwaError, if any. Returns NO_ERROR if memory 
     * allocation is successful, or an appropriate error code otherwise.
     */
    DiwaError initializeWeights();

    /**
     * @brief Tests the inference of the neural network for a given input.
     *
     * This function tests the output of the neural network for a given input
     * against the expected output. It checks whether the inferred output matches
     * the expected output for each output neuron.
     *
     * @param testInput Pointer to the input values for testing.
     * @param testExpectedOutput Pointer to the expected output values for testing.
     * @return True if the inferred output matches the expected output for all output neurons, false otherwise.
     */
    bool testInference(double *testInput, double *testExpectedOutput);

public:
    /**
     * @brief Default constructor for the Diwa class.
     *
     * This constructor initializes a new instance of the Diwa class.
     * It sets up the neural network with default value 0 on parameters.
     */
    Diwa();

    /**
     * @brief Destructor for the Diwa class.
     *
     * This destructor releases resources associated with the Diwa object
     * upon its destruction. It ensures proper cleanup to prevent memory leaks.
     */
    ~Diwa();

    /**
     * @brief Initializes the Diwa neural network with specified parameters.
     *
     * This method initializes the Diwa neural network with the given parameters,
     * including the number of input neurons, hidden layers, hidden neurons per layer,
     * and output neurons. Additionally, it allows the option to randomize the weights
     * in the network if desired.
     *
     * @param inputNeurons Number of input neurons in the neural network.
     * @param hiddenLayers Number of hidden layers in the neural network.
     * @param hiddenNeurons Number of neurons in each hidden layer.
     * @param outputNeurons Number of output neurons in the neural network.
     * @param randomizeWeights Flag indicating whether to randomize weights in the network (default is true).
     * 
     * @return DiwaError indicating the initialization status.
     */
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

    /**
     * @brief Load neural network model from file in Arduino environment.
     *
     * This method loads a previously saved neural network model from the specified file
     * in an Arduino environment. It reads the model data from the given file and initializes
     * the Diwa object with the loaded model parameters and weights.
     *
     * @param annFile File object representing the neural network model file.
     * @return DiwaError indicating the loading status.
     */
    DiwaError loadFromFile(File annFile);

    /**
     * @brief Save neural network model to file in Arduino environment.
     *
     * This method saves the current state of the neural network model to the specified
     * file in an Arduino environment. It writes the model parameters and weights to the
     * given file, allowing later retrieval and reuse of the trained model.
     *
     * @param annFile File object representing the destination file for the model.
     * @return DiwaError indicating the saving status.
     */
    DiwaError saveToFile(File annFile);

    #else

    /**
     * @brief Load neural network model from file in non-Arduino environment.
     *
     * This method loads a previously saved neural network model from the specified file
     * in a non-Arduino environment. It reads the model data from the given file stream
     * and initializes the Diwa object with the loaded model parameters and weights.
     *
     * @param annFile Input file stream representing the neural network model file.
     * @return DiwaError indicating the loading status.
     */
    DiwaError loadFromFile(std::ifstream& annFile);

    /**
     * @brief Save neural network model to file in non-Arduino environment.
     *
     * This method saves the current state of the neural network model to the specified
     * file in a non-Arduino environment. It writes the model parameters and weights to
     * the given file stream, facilitating storage and retrieval of the trained model.
     *
     * @param annFile Output file stream representing the destination file for the model.
     * @return DiwaError indicating the saving status.
     */
    DiwaError saveToFile(std::ofstream& annFile);

    #endif

    /**
     * @brief Calculates the accuracy of the neural network on test data.
     *
     * This function calculates the accuracy of the neural network on a given
     * set of test data. It compares the inferred output with the expected output
     * for each test sample and calculates the percentage of correct inferences.
     *
     * @param testInput Pointer to the input values of the test data.
     * @param testExpectedOutput Pointer to the expected output values of the test data.
     * @param epoch Total number of test samples in the test data.
     * 
     * @return The accuracy of the neural network on the test data as a percentage.
     */
    double calculateAccuracy(double *testInput, double *testExpectedOutput, int epoch);

    /**
     * @brief Calculates the loss of the neural network on test data.
     *
     * This function calculates the loss of the neural network on a given set
     * of test data. It computes the percentage of test samples for which the
     * inferred output does not match the expected output.
     *
     * @param testInput Pointer to the input values of the test data.
     * @param testExpectedOutput Pointer to the expected output values of the test data.
     * @param epoch Total number of test samples in the test data.
     * 
     * @return The loss of the neural network on the test data as a percentage.
     */
    double calculateLoss(double *testInput, double *testExpectedOutput, int epoch);

    /**
     * @brief Sets the activation function for the neural network.
     *
     * This method allows the user to set the activation function used by the neurons in the neural network.
     * The activation function determines the output of a neuron based on its input. Different activation
     * functions can be used depending on the nature of the problem being solved and the characteristics of
     * the dataset. Common activation functions include sigmoid, ReLU, and tanh.
     *
     * @param activation The activation function to be set for the neural network.
     * @see Diwa::getActivationFunction()
     */
    void setActivationFunction(diwa_activation activation);

    /**
     * @brief Retrieves the current activation function used by the neural network.
     *
     * This method returns the activation function currently set for the neurons in the neural network.
     * It allows the user to query the current activation function being used for inference and training
     * purposes. The activation function determines the output of a neuron based on its input. Different
     * activation functions can be used depending on the nature of the problem being solved and the
     * characteristics of the dataset. Common activation functions include sigmoid, ReLU, and tanh.
     *
     * @return The activation function currently set for the neural network.
     * @see Diwa::setActivationFunction()
     */
    diwa_activation getActivationFunction() const;

    /**
     * @brief Calculates the recommended number of hidden neurons based on the input and output neurons.
     *
     * This function computes the recommended number of hidden neurons for a neural network 
     * based on the number of input and output neurons. The recommendation is calculated 
     * using a heuristic formula that aims to strike a balance between model complexity 
     * and generalization ability. The recommended number of hidden neurons is determined 
     * as the square root of the product of the input and output neurons.
     *
     * @return The recommended number of hidden neurons, or -1 if the input or output neurons are non-positive.
     */
    int recommendedHiddenNeuronCount();

    /**
     * @brief Calculates the recommended number of hidden layers based on the dataset size and complexity.
     *
     * This function computes the recommended number of hidden layers for a neural network 
     * based on the size and complexity of the dataset. The recommendation is calculated 
     * using a heuristic formula that takes into account the number of samples, input neurons, 
     * output neurons, and a scaling factor alpha. The recommended number of hidden layers 
     * is determined as the total number of samples divided by (alpha times the sum of input 
     * and output neurons).
     *
     * @param numSamples The total number of samples in the dataset.
     * @param alpha A scaling factor used to adjust the recommendation based on dataset complexity.
     * 
     * @return The recommended number of hidden layers, or -1 if any of the input parameters are non-positive.
     */
    int recommendedHiddenLayerCount(int numSamples, int alpha);
};

#endif  // DIWA_H