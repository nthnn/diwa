<p align="center">
    <img src="https://github.com/nthnn/diwa/blob/main/logo.png" width="200" />
    <br/><br/>
    <h1>Diwa: Arduino Tiny AI/ML Library</h1>
</p>

![Arduino Release](https://img.shields.io/badge/Library%20Manager-v0.0.1-red?logo=Arduino)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nthnn/diwa/blob/main/LICENSE)

Diwa is a lightweight library providing a simple implementation of Feedforward Artificial Neural Networks (ANNs) for microcontrollers such as ESP8266, ESP32, and similar development boards. It is designed for resource-constrained environments, offering a streamlined solution for tasks that require neural network capabilities.

Diwa stands out as a straightforward and effective solution for implementing artificial neural networks on microcontrollers. Key features include:

- **Lightweight**: Designed for resource-constrained microcontroller environments.
- **Simple Implementation**: Provides a basic yet effective implementation of a Feedforward ANN.
- **Easy Integration**: Suitable for microcontrollers like ESP8266, ESP32, and similar boards.
- **Training Support**: Includes methods for training the neural network using backpropagation.

> [!NOTE]
> Diwa is primarily intended for lightweight applications. For more intricate tasks, consider using advanced machine learning libraries on more powerful platforms.

See live demo on [Wokwi](https://wokwi.com/projects/387551593748039681).

## Getting Started

To start using Diwa library in your Arduino projects, follow these simple steps:

1. Download the Diwa library from the GitHub repository.
2. Extract the downloaded archive and rename the folder to "diwa".
3. Move the "diwa" folder to the Arduino libraries directory on your computer.
    - Windows: `Documents\Arduino\libraries\`
    - MacOS: `~/Documents/Arduino/libraries/`
    - Linux: `~/Arduino/libraries/`
4. Launch the Arduino IDE.

## Examples

To access the examples:

1. Open the Arduino IDE.
2. Click on `File > Examples > diwa` to see the list of available examples.
3. Upload the example sketch to your Arduino board and see the results in action.

Here's a full example usage.
```cpp
#include <diwa.hpp>

void setup() {
    // Initialize serial communication with a baud rate of 115200
    Serial.begin(115200);

    // Define training input and output data for XOR operation
    float trainingInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float trainingOutput[4][1] = {{0}, {1}, {1}, {0}};

    // Create an instance of the Diwa neural network with 2 input neurons, 1 hidden layer with 3 neurons, and 1 output neuron
    Diwa network(2, 1, 3, 1);

    // Train the network for 3000 epochs using the XOR training data
    for(int epoch = 0; epoch < 3000; epoch++) {
        // Train the network for each set of input and target output values
        network.train(4, trainingInput[0], trainingOutput[0]);
        network.train(4, trainingInput[1], trainingOutput[1]);
        network.train(4, trainingInput[2], trainingOutput[2]);
        network.train(4, trainingInput[3], trainingOutput[3]);
    }

    // Perform inference on the trained network and print the results
    for(int i = 0; i < 4; i++) {
        // Get the current input row
        float* row = trainingInput[i];

        // Perform inference using the trained network
        float* inferred = network.inference(row);

        // Print the result for the current input
        char str[100];
        sprintf(str, "Output for [%1.f, %1.f]: %1.f (%g)\n", row[0], row[1], inferred[0], inferred[0]);
        Serial.print(str);
    }
}

void loop() { }
```

## Contribution and Feedback

Contributions and feedback are all welcome to enhance this library. If you encounter any issues, have suggestions for improvements, or would like to contribute code, please do so.

## API Reference

```cpp
Diwa(
    int inputNeurons,
    int hiddenLayers,
    int hiddenNeurons,
    int outputNeurons
);
```

**Description:**

Creates an instance of the Diwa class with the specified number of input neurons, hidden layers, hidden neurons, and output neurons.

**Parameters:**

- `inputNeurons` - Number of input neurons.
- `hiddenLayers` - Number of hidden layers.
- `hiddenNeurons` - Number of neurons in each hidden layer.
- `outputNeurons` - Number of output neurons.

---

```cpp
float* Diwa::inference(float *inputs);
```

**Description:**

Given an array of input values, this method computes and returns an array of output values through the neural network.

**Parameters:**

- `inputs` - Array of input values for the neural network.

**Returns:**

Array of output values after inference.

---

```cpp
void Diwa::train(
    float learningRate,
    float *inputNeurons,
    float *outputNeurons
);
```

**Description:**

This method facilitates the training of the neural network by adjusting its weights based on the provided input and target output values.

**Parameters:**

- `learningRate` - Learning rate for the training process.
- `inputNeurons` - Array of input values for training.
- `outputNeurons` - Array of target output values for training.

## License

Copyright 2023 - Nathanne Isip

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.