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
#include <diwa.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    // Create an instance of the Diwa neural network
    Diwa network;

    // Define input-output pairs for training the neural network
    double trainingInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double trainingOutput[4][1] = {{1}, {0}, {0}, {1}};

    // Initialize the neural network with specified parameters
    if(network.initialize(2, 1, 3, 1) != NO_ERROR) {  // Check if initialization was successful
        cout << "Failed to initialize neural network" << endl;
        exit(0);  // Exit the program
    }

    // Train the neural network for a certain number of epochs
    for(int epoch = 0; epoch < 5000; epoch++) {
        // Iterate through each input-output pair and train the network
        network.train(6, trainingInput[0], trainingOutput[0]);
        network.train(6, trainingInput[1], trainingOutput[1]);
        network.train(6, trainingInput[2], trainingOutput[2]);
        network.train(6, trainingInput[3], trainingOutput[3]);
    }

    // Perform inference for each input and print the output
    for(uint8_t i = 0; i < 4; i++) {
        double* row = trainingInput[i];  // Get the current input row
        double* inferred = network.inference(row);  // Perform inference using the neural network

        // Print the output for the current input
        cout << "Output for [" << fixed << setprecision(1) << row[0] << ", "
              << fixed << setprecision(1) << row[1] << "]: "
              << fixed << setprecision(1) << inferred[0] << " ("
              << scientific << inferred[0] << ")" << endl;
    }

    return 0;
}