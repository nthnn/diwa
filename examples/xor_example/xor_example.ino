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

void setup() {
    // Initialize serial communication with a baud rate of 115200
    Serial.begin(115200);

    // Define training input and output data for XOR operation
    double trainingInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double trainingOutput[4][1] = {{1}, {0}, {0}, {1}};

    // Create an instance of the Diwa neural network with 2 input neurons,
    // 1 hidden layer with 3 neurons, and 1 output neuron
    Diwa network;
    network.initialize(2, 1, 3, 1);

    // Train the network for 3000 epochs using the XOR training data
    for(uint32_t epoch = 0; epoch < 10000; epoch++) {
        // Train the network for each set of input and target output values
        network.train(6, trainingInput[0], trainingOutput[0]);
        network.train(6, trainingInput[1], trainingOutput[1]);
        network.train(6, trainingInput[2], trainingOutput[2]);
        network.train(6, trainingInput[3], trainingOutput[3]);

        // Show accuracy and loss on training for every 100th epoch
        if(epoch % 500 == 0) {
            double accuracy = 0.0, loss = 0.0;

            // Calculate accuracy and loss for each training sample
            for(uint8_t i = 0; i < 4; i++) {
                accuracy += network.calculateAccuracy(trainingInput[i], trainingOutput[i], 3);
                loss += network.calculateLoss(trainingInput[i], trainingOutput[i], 3);
            }

            // Average accuracy and loss over all samples
            accuracy /= 4, loss /= 4;

            Serial.print("Epoch: ");
            Serial.print(epoch);
            Serial.print(", Accuracy: ");
            Serial.print(accuracy * 100);
            Serial.print("%, Loss: ");
            Serial.print(loss * 100);
            Serial.println("%");
        }
    }

    // Perform inference on the trained network and print the results
    for(uint8_t i = 0; i < 4; i++) {
        // Get the current input row
        double* row = trainingInput[i];

        // Perform inference using the trained network
        double* inferred = network.inference(row);

        // Print the result for the current input
        char str[100];
        sprintf(str, "Output for [%1.f, %1.f]: %1.f (%g)\n", row[0], row[1], inferred[0], inferred[0]);
        Serial.print(str);
    }
}

void loop() {
  vTaskDelay(10);
}