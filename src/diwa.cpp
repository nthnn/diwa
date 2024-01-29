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

#include <bootloader_random.h>
#include <diwa.hpp>

static float diwaSigmoid(float value) {
    if(value < -45.0) return 0;
    if(value > 45.0) return 1;

    return 1.0 / (1.0 + exp(-value));
}

Diwa::Diwa(int inputNeurons, int hiddenLayers, int hiddenNeurons, int outputNeurons) {
    assert(ESP.getFreePsram() != 0);
    assert(!(hiddenLayers < 0 || inputNeurons < 0 || outputNeurons < 1 ||
        (hiddenLayers > 0 && hiddenNeurons < 1)));

    bootloader_random_enable();
    randomSeed(esp_random());
    bootloader_random_disable();

    const int hiddenWeightCount = hiddenLayers ?
        (inputNeurons + 1) * hiddenNeurons +
            (hiddenLayers - 1) * (hiddenNeurons + 1) *
            hiddenNeurons : 0;
    const int outputWeightCount = (hiddenLayers ?
        (hiddenNeurons + 1) : (inputNeurons + 1)
    ) * outputNeurons;

    const int weightCount = hiddenWeightCount + outputWeightCount;
    const int neuronCount = inputNeurons + hiddenNeurons * hiddenLayers + outputNeurons;

    this->inputNeurons = inputNeurons;
    this->hiddenLayers = hiddenLayers;
    this->hiddenNeurons = hiddenNeurons;
    this->outputNeurons = outputNeurons;

    this->weightCount = weightCount;
    this->neuronCount = neuronCount;

    this->weights = (float*) ps_malloc(sizeof(float) * (weightCount * neuronCount));
    this->outputs = this->weights + this->weightCount;
    this->deltas = this->outputs + this->neuronCount;

    this->randomizeWeights();
}

void Diwa::randomizeWeights() {
    for(int i = 0; i < this->weightCount; i++)
        this->weights[i] = (((float) random()) / RAND_MAX) - 0.5;
}

float* Diwa::inference(float *inputNeurons) {
    float *weights = this->weights;
    float *inputs = this->outputs;
    float *outputs = this->outputs + this->inputNeurons;

    memcpy(this->outputs, inputNeurons, sizeof(float) * this->inputNeurons);
    if(!this->hiddenLayers) {
        float *returnValues = outputs;
        for(int j = 0; j < this->outputNeurons; ++j) {
            float sum = *weights++ * -1.0;

            for(int k = 0; k < this->inputNeurons; ++k)
                sum += *weights++ * inputs[k];
            *outputs++ = diwaSigmoid(sum);
        }

        return returnValues;
    }

    for(int j = 0; j < this->hiddenNeurons; ++j) {
        float sum = *weights++ * -1.0;

        for(int k = 0; k < this->inputNeurons; ++k)
            sum += *weights++ * inputs[k];
        *outputs++ = diwaSigmoid(sum);
    }

    inputs += this->inputNeurons;
    for(int h = 1; h < this->hiddenLayers; ++h) {
        for(int j = 0; j < this->hiddenNeurons; ++j) {
            float sum = *weights++ * -1.0;
            
            for(int k = 0; k < this->hiddenNeurons; ++k)
                sum += *weights++ * inputs[k];
            *outputs++ = diwaSigmoid(sum);
        }

        inputs += this->hiddenNeurons;
    }

    float* returnValue = outputs;
    for(int j = 0; j < this->outputNeurons; ++j) {
        float sum = *weights++ * -1.0;

        for(int k = 0; k < this->hiddenNeurons; ++k)
            sum += *weights++ * inputs[k];
        *outputs++ = diwaSigmoid(sum);;
    }

    return returnValue;
}

void Diwa::train(float learningRate, float *inputNeurons, float *outputNeurons) {
    this->inference(inputNeurons);

    {
        float *outputs =
            this->outputs +
            this->inputNeurons +
            this->hiddenNeurons *
            this->hiddenLayers;
        float *deltas =
            this->deltas +
            this->hiddenNeurons *
            this->hiddenLayers;
        float *training = outputNeurons;

        for(int j = 0; j < this->outputNeurons; ++j) {
            *deltas++ = (*training - *outputs) *
                *outputs * (1.0 - *outputs);
            outputs++, training++;
        }
    }

    for(int h = this->hiddenLayers - 1; h >= 0; --h) {
        float *outputs =
            this->outputs +
            this->inputNeurons +
            (h * this->hiddenNeurons);

        float *deltas =
            this->deltas +
            (h * this->hiddenNeurons);

        float *firstDelta =
            this->deltas +
            ((h + 1) * this->hiddenNeurons);

        float *firstWeight =
            this->weights +
            ((this->inputNeurons + 1) * this->hiddenNeurons) +
            ((this->hiddenNeurons + 1) * this->hiddenNeurons * h);

        for(int j = 0; j < this->hiddenNeurons; ++j) {
            float delta = 0;

            for(int k = 0;
                k < (h == this->hiddenLayers - 1 ?
                    this->outputNeurons :
                    this->hiddenNeurons);
                ++k) {
                int weightIndex = k * (this->hiddenNeurons + 1) + (j + 1);

                float forwardDelta = firstDelta[k];
                float forwardWeight = firstWeight[weightIndex];

                delta += forwardDelta * forwardWeight;
            }

            *deltas = *outputs * (1.0 - *outputs) * delta;
            deltas++, outputs++;
        }
    }

    {
        float *deltas =
            this->deltas +
            this->hiddenNeurons *
            this->hiddenLayers;

        float *weights =
            this->weights +
            (this->hiddenLayers ?
                (this->inputNeurons + 1) * this->hiddenNeurons +
                (this->hiddenNeurons + 1) * this->hiddenNeurons *
                (this->hiddenLayers - 1) : 0);

        float *firstOutput =
            this->outputs +
            (this->hiddenLayers ?
                (this->inputNeurons + this->hiddenNeurons *
                    (this->hiddenLayers - 1)) : 0);

        for(int j = 0; j < this->outputNeurons; ++j) {
            *weights++ += *deltas * learningRate * -1.0;

            for(int k = 1;
                k < (this->hiddenLayers ?
                    this->hiddenNeurons : this->inputNeurons) + 1;
                ++k)
                *weights++ += *deltas * learningRate * firstOutput[k - 1];

            deltas++;
        }
    }

    for(int h = this->hiddenLayers - 1; h >= 0; --h) {
        float *deltas = this->deltas +
            (h * this->hiddenNeurons);

        float *firstInput = this->outputs +
            (h ? this->inputNeurons +
                this->hiddenNeurons * (h - 1) : 0);

        float *weights =
            this->weights + (h ?
                (this->inputNeurons + 1) * this->hiddenNeurons +
                (this->hiddenNeurons + 1) * this->hiddenNeurons *
                (h - 1) : 0);

        for(int j = 0; j < this->hiddenNeurons; ++j) {
            *weights += *deltas * learningRate * -1.0;

            for(int k = 1;
                k < (h == 0 ?
                    this->inputNeurons : this->hiddenNeurons) + 1;
                ++k)
                *weights++ += *deltas * learningRate * firstInput[k - 1];

            deltas++;
        }
    }
}