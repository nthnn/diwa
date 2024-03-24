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

// CS pin definition for SD card
#define SD_CS_PIN 5

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
    if(network.initialize(2, 1, 3, 1) != NO_ERROR) {
        Serial.println(F("Something went wrong initializing neural network."));
        while(true);
    }

    // Train the neural network
    Serial.println(F("Training neural network... "));
    for(int epoch = 0; epoch <= 7000; epoch++) {
        network.train(6, trainingInput[0], trainingOutput[0]);
        network.train(6, trainingInput[1], trainingOutput[1]);
        network.train(6, trainingInput[2], trainingOutput[2]);
        network.train(6, trainingInput[3], trainingOutput[3]);

        // Show accuracy and loss on training for every 100th epoch
        if((epoch % 1000 == 0) || epoch == 7000) {
            double accuracy = 0.0, loss = 0.0;

            // Calculate accuracy and loss for each training sample
            for(uint8_t i = 0; i < 4; i++) {
                accuracy += network.calculateAccuracy(trainingInput[i], trainingOutput[i], 3);
                loss += network.calculateLoss(trainingInput[i], trainingOutput[i], 3);
            }

            // Average accuracy and loss over all samples
            accuracy /= 4, loss /= 4;

            // Print the accuracy and loss
            Serial.print(F("Epoch: "));
            Serial.print(epoch);
            Serial.print(F("\t| Accuracy: "));
            Serial.print(accuracy * 100);
            Serial.print(F("%\t| Loss: "));
            Serial.print(loss * 100);
            Serial.println(F("%"));
        }
    }
    Serial.println(F("Training done!\r\n"));

    // Test inferences
    Serial.println(F("Testing inferences... "));
    for(uint8_t i = 0; i < 4; i++) {
        double* row = trainingInput[i];
        double* inferred = network.inference(row);

        // Print the result for the current input
        char str[100];
        sprintf(str, "\t[%g, %g]: %d (%g)\n", row[0], row[1], (inferred[0] >= 0.5), inferred[0]);
        Serial.print(str);
    }

    // Save trained model to file
    Serial.print(F("Saving trained model to file... "));
    network.saveToFile(outfile);

    outfile.close();
    Serial.println(F("done!\r\n"));
}

// Function to load a trained model from a file and perform inferences
void loadAndRead() {
    // Create a Diwa object
    Diwa network;

    // Open the saved model file for reading
    File infile = SD.open("/model.ann", "r");
    Serial.println(F("Loading model file..."));

    // Load the trained model from the file
    if(network.loadFromFile(infile) != NO_ERROR) {
        Serial.println(F("Something went wrong loading model file."));
        while(true);
    }
    else Serial.println(F("Neural network model successfully loaded!"));

    // Close the input trained model file
    infile.close();

    double testInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Serial.println(F("Testing inferences... "));

    // Test inferences with new input data
    for(uint8_t i = 0; i < 4; i++) {
        double* row = testInput[i];
        double* inferred = network.inference(row);

        // Print the result for the current input
        char str[100];
        sprintf(str, "\t[%g, %g]: %d (%g)\n", row[0], row[1], (inferred[0] >= 0.5), inferred[0]);
        Serial.print(str);
    }
}

// Setup function to demonstrate training, saving, loading, and inference
void setup() {
    // Initialize serial communication
    Serial.begin(115200);

    // Initialize the SD card connected to ESP32 via SPI
    if(!SD.begin(SD_CS_PIN)) {
        Serial.println(F("Something went wrong initializing SD card."));
        while(true);
    }

    #if defined(ARDUINO_ARCH_ESP32)
    // Check the ESP32 PSRAM to initialize
    if(psramFound() && !psramInit()) {
        Serial.println(F("Cannot initialize PSRAM."));
        while(true);
    }
    #endif

    trainAndSave(); // Train the neural network and save the trained model
    loadAndRead();  // Load the trained model and perform inference
}

void loop() {
    delay(100);
}