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
#include <iomanip>
#include <iostream>

using namespace std;

// Function to train the neural network and save the trained model to a file
void trainAndSave() {
    // Create a Diwa object
    Diwa network;

    // Open a file for writing the trained model
    ofstream outfile("model.ann", ios::binary);

    // Training data
    double trainingInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double trainingOutput[4][1] = {{1}, {0}, {0}, {1}};

    // Initialize the neural network with specified parameters
    DiwaError error;
    if((error = network.initialize(2, 1, 3, 1)) == NO_ERROR)
        cout << "Done initializing neural network." << endl;
    else {
        cout << "Something went wrong initializing neural network." << endl;
        exit(0);
    }

    // Train the neural network
    cout << "Training neural network... ";
    for(int epoch = 0; epoch < 5000; epoch++) {
        network.train(6, trainingInput[0], trainingOutput[0]);
        network.train(6, trainingInput[1], trainingOutput[1]);
        network.train(6, trainingInput[2], trainingOutput[2]);
        network.train(6, trainingInput[3], trainingOutput[3]);
    }
    cout << "done!" << endl;

    // Test inferences
    cout << "Testing inferences... " << endl;
    for(uint8_t i = 0; i < 4; i++) {
        double* row = trainingInput[i];
        double* inferred = network.inference(row);

        cout << "Output for [" << fixed << setprecision(1) << row[0] << ", "
              << fixed << setprecision(1) << row[1] << "]: "
              << fixed << setprecision(1) << inferred[0] << " ("
              << scientific << inferred[0] << ")" << endl;
    }

    // Save trained model to file
    cout << "Saving trained model to file... ";
    network.saveToFile(outfile);

    outfile.close();
    cout << "done!" << endl;
}

// Function to load a trained model from a file and perform inferences
void loadAndRead() {
    // Create a Diwa object
    Diwa network;

    // Open the saved model file for reading
    ifstream infile("model.ann", ios::binary);

    // Load the trained model from the file
    DiwaError error;
    if((error = network.loadFromFile(infile)) == NO_ERROR)
        cout << "Model loaded successfully!" << endl;
    else {
        cout << "Something went wrong loading model file." endl;
        exit(0);
    }

    // Close the input trained model file
    infile.close();

    double testInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    cout << "Testing inferences... " << endl;

    // Test inferences with new input data
    for(uint8_t i = 0; i < 4; i++) {
        double* row = testInput[i];
        double* inferred = network.inference(row);

        cout << "Output for [" << fixed << setprecision(1) << row[0] << ", "
              << fixed << setprecision(1) << row[1] << "]: "
              << fixed << setprecision(1) << inferred[0] << " ("
              << scientific << inferred[0] << ")" << endl;
    }
}

// Main function to demonstrate training, saving, loading, and inference
int main() {
    trainAndSave(); // Train the neural network and save the trained model
    loadAndRead();  // Load the trained model and perform inference

    return 0;
}