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
#include <SD.h>

// Function to train the neural network and save the trained model to a file
void trainAndSave() {
    // Create a Diwa object
    Diwa network;

    // Open a file for writing the trained model
    File outfile = SD.open("/model.ann", "w");

    // Training data
    double trainingInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double trainingOutput[4][1] = {{1}, {0}, {0}, {1}};

    // Initialize the neural network with specified parameters
    if(network.initialize(2, 1, 3, 1) == NO_ERROR)
        Serial.println("Done initializing neural network.");
    else {
        Serial.println("Something went wrong initializing neural network.");
        while(true);
    }

    // Train the neural network
    Serial.print("Training neural network... ");
    for(int epoch = 0; epoch < 5000; epoch++) {
        network.train(6, trainingInput[0], trainingOutput[0]);
        network.train(6, trainingInput[1], trainingOutput[1]);
        network.train(6, trainingInput[2], trainingOutput[2]);
        network.train(6, trainingInput[3], trainingOutput[3]);
    }
    Serial.println("done!");

    // Test inferences
    Serial.println("Testing inferences... ");
    for(uint8_t i = 0; i < 4; i++) {
        double* row = trainingInput[i];
        double* inferred = network.inference(row);

        // Print the result for the current input
        char str[100];
        sprintf(str, "Output for [%1.f, %1.f]: %1.f (%g)\n", row[0], row[1], inferred[0], inferred[0]);
        Serial.print(str);
    }

    // Save trained model to file
    Serial.print("Saving trained model to file... ");
    network.saveToFile(outfile);

    outfile.close();
    Serial.println("done!");
}

// Function to load a trained model from a file and perform inferences
void loadAndRead() {
    // Create a Diwa object
    Diwa network;

    // Open the saved model file for reading
    File infile = SD.open("/model.ann", "r");

    // Load the trained model from the file
    if(network.loadFromFile(infile) == NO_ERROR)
        Serial.println("Model loaded successfully!");
    else {
        Serial.println("Something went wrong loading model file.");
        while(true);
    }

    // Close the input trained model file
    infile.close();

    double testInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Serial.println("Testing inferences... ");

    // Test inferences with new input data
    for(uint8_t i = 0; i < 4; i++) {
        double* row = testInput[i];
        double* inferred = network.inference(row);

        // Print the result for the current input
        char str[100];
        sprintf(str, "Output for [%1.f, %1.f]: %1.f (%g)\n", row[0], row[1], inferred[0], inferred[0]);
        Serial.print(str);
    }
}

// Setup function to demonstrate training, saving, loading, and inference
void setup() {
    // Initialize serial communication
    Serial.begin(115200);

    // Initialize the SD card connected to ESP32 via SPI
    if(!SD.begin(5)) {
        Serial.println("Something went wrong initializing SD card.");
        while(true);
    }

    // Check the ESP32 PSRAM to initialize
    if(!psramInit()) {
        Serial.println("Cannot initialize PSRAM.");
        while(true);
    }

    trainAndSave(); // Train the neural network and save the trained model
    loadAndRead();  // Load the trained model and perform inference
}

void loop() {
  vTaskDelay(10);
}