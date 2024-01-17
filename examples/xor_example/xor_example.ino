#include <diwa.hpp>

void setup() {
    Serial.begin(115200);

    float trainingInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float trainingOutput[4][1] = {{0}, {1}, {1}, {0}};

    Diwa network(2, 1, 3, 1);
    for(int epoch = 0; epoch < 3000; epoch++) {
        network.train(4, trainingInput[0], trainingOutput[0]);
        network.train(4, trainingInput[1], trainingOutput[1]);
        network.train(4, trainingInput[2], trainingOutput[2]);
        network.train(4, trainingInput[3], trainingOutput[3]);
    }

    for(int i = 0; i < 4; i++) {
        float* row = trainingInput[i];
        float* inferred = network.inference(row);

        char str[100];
        sprintf(str, "Output for [%1.f, %1.f]: %1.f (%g)\n", row[0], row[1], inferred[0], inferred[0]);
        Serial.print(str);
    }
}

void loop() { }