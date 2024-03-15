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

    #ifdef ARDUINO
    NO_ESP_PSRAM            /**< ESP PSRAM not available */
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
     * 
     * @brief Randomize the weights in the network.
     * 
     */
    void randomizeWeights();

    /**
     * 
     * @brief Allocates memory for neural network weights.
     * 
     * @return DiwaError The corresponding DiwaError, if any.
    */
    DiwaError initializeWeights();

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
};

#endif  // DIWA_H