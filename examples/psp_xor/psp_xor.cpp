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
#include <pspdebug.h>

PSP_MODULE_INFO("XOR Example", 0, 1, 0);

int main() {
    // Initialize PSP debug screen
    pspDebugScreenInit();

    // Define training input and output data for XOR operation
    double trainingInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double trainingOutput[4][1] = {{1}, {0}, {0}, {1}};

    // Create an instance of the Diwa neural network with 2 input neurons,
    // 1 hidden layer with 3 neurons, and 1 output neuron
    Diwa network;
    if(network.initialize(2, 1, 3, 1) != NO_ERROR) {
        pspDebugScreenPrintf("Something went wrong initializing neural network.");
        while(true);
    }

    // Train the network for 3000 epochs using the XOR training data
    pspDebugScreenPrintf("Starting training...\r\n");
    for(uint32_t epoch = 0; epoch <= 5000; epoch++) {
        // Train the network for each set of input and target output values
        network.train(6, trainingInput[0], trainingOutput[0]);
        network.train(6, trainingInput[1], trainingOutput[1]);
        network.train(6, trainingInput[2], trainingOutput[2]);
        network.train(6, trainingInput[3], trainingOutput[3]);

        // Show accuracy and loss on training for every 100th epoch
        if((epoch % 1000 == 0) || epoch == 5000) {
            double accuracy = 0.0, loss = 0.0;

            // Calculate accuracy and loss for each training sample
            for(uint8_t i = 0; i < 4; i++) {
                accuracy += network.calculateAccuracy(trainingInput[i], trainingOutput[i], 3);
                loss += network.calculateLoss(trainingInput[i], trainingOutput[i], 3);
            }

            // Average accuracy and loss over all samples
            accuracy /= 4, loss /= 4;
            // Get the percentage of each calculated average
            accuracy *= 100, loss *= 100;

            // Print the accuracy and loss per epoch
            if(epoch == 0)
                pspDebugScreenPrintf("Epoch: 0   \t");
            else pspDebugScreenPrintf("Epoch: %d\t", epoch);
            pspDebugScreenPrintf("| Accuraccy: %g%%\t| Loss: %g%%\r\n", accuracy, loss);
        }
    }
    pspDebugScreenPrintf("Training done!\r\n\r\n");

    // Perform inference on the trained network and print the results
    pspDebugScreenPrintf("Testing inferences...\r\n");
    for(uint8_t i = 0; i < 4; i++) {
        // Get the current input row
        double* row = trainingInput[i];

        // Perform inference using the trained network
        double* inferred = network.inference(row);

        // Print the result for the current input
        pspDebugScreenPrintf("\t[%g, %g]: %d (%g)\r\n", row[0], row[1], (inferred[0] >= 0.5), inferred[0]);
    }

    return 0;
}