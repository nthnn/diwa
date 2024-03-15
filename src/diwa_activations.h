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
 * @file diwa_activations.h
 * @author [Nathanne Isip](https://github.com/nthnn)
 * @brief Defines activation functions for use in the Diwa neural network.
 *
 * This header file defines the DiwaActivationFunc class, which provides a set of common activation functions
 * used in neural networks. Activation functions play a crucial role in determining the output of a neuron
 * based on its input. They introduce non-linearity to the network, allowing it to learn complex patterns
 * and relationships in the data.
 *
 * The DiwaActivationFunc class contains static methods for popular activation functions, including sigmoid
 * and gaussian functions. These activation functions are used to introduce non-linearity and regulate
 * the output values of neurons in the neural network. They transform the input values into output values
 * suitable for the task at hand, such as classification or regression.
 *
 * @note Activation functions are an essential component of neural networks and significantly influence
 *       the network's learning dynamics and performance. The choice of activation function depends on
 *       the nature of the problem being solved and the characteristics of the dataset.
 */

#ifndef DIWA_ACTIVATIONS_H
#define DIWA_ACTIVATIONS_H

#include <math.h>

#define DIWA_ACTFUNC_LOWER_BOUND -30.0f /**< Lower bound for input values to prevent overflow. */
#define DIWA_ACTFUNC_UPPER_BOUND 30.0f  /**< Upper bound for input values to prevent overflow. */

/**
 * @brief Typedef for activation function pointer.
 *
 * This typedef defines the signature for activation functions used in the Diwa neural network.
 * Activation functions take a single input value and return a transformed output value.
 */
typedef double (*diwa_activation)(double);

/**
 * @brief Class containing static methods for common activation functions.
 *
 * The DiwaActivationFunc class provides a set of static methods for common activation functions
 * used in neural networks. These activation functions transform the input value to produce the
 * output value of a neuron. Supported activation functions include sigmoid and gaussian functions.
 */
class DiwaActivationFunc final {
public:
    /**
     * @brief Sigmoid activation function.
     *
     * The sigmoid activation function takes an input value and returns the corresponding output
     * value after applying the sigmoid transformation. It ensures that the output value is bounded
     * between 0 and 1, suitable for binary classification tasks and preventing overflow.
     *
     * @param x The input value to be transformed.
     * @return The transformed output value after applying the sigmoid function.
     */
    static inline double sigmoid(double x) {
        if(x < DIWA_ACTFUNC_LOWER_BOUND)
            return 0;
        if(x > DIWA_ACTFUNC_UPPER_BOUND)
            return 1;

        return 1.0 / (1.0 + exp(-x));
    }

    /**
     * @brief Gaussian activation function.
     *
     * The gaussian activation function takes an input value and returns the corresponding output
     * value after applying the gaussian transformation. It produces a bell-shaped curve, ensuring
     * that the output value decreases smoothly as the input value moves away from the center.
     *
     * @param x The input value to be transformed.
     * @return The transformed output value after applying the gaussian function.
     */
    static inline double gaussian(double x) {
        if(x < DIWA_ACTFUNC_LOWER_BOUND)
            return 0;
        if(x > DIWA_ACTFUNC_UPPER_BOUND)
            return 1;

        return 1.0 / exp(x * x);
    }
};

#endif