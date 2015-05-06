//
//  neuralnetwork.cpp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 10/15/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#include "network.hpp"

namespace NNetwork {
    

    // initialize weights
    void Network::initialize(void) {

        // Because Eigen uses srand
        srand(static_cast<unsigned int>( time(nullptr) ));
        
        // topology parameters
        _numInputs = _topology[0];
        _numOutputs = _topology[_topology.size() - 1];
        _numLayers = _topology.size();
        
        // weights
        scaleWeightVectors();
        
        // activation, gradient functions per layer
        _functions.resize(_numLayers);
        
        setupDefaultNetwork();
        
        // Not a softmax activation at the output
        _isSoftmax = false;
        
    }
    

    void Network::setupDefaultNetwork(){
        
        // random weights
        setRandomWeights();
        
        for (int i = 1; i < _numLayers; i++)
            _functions[i] = FunctionalitySuite::logistic();
            
        // For input layer
        setFunctionalityAtLayer(0,
                                *FunctionalitySuite::customSuite([](double x)->double { return x; },
                                                                 [](double)->double {return 0;},
                                                                 [] (double, double)->double {return 0; }));
        
    }
    
    
    void Network::scaleWeightVectors(){
        _weights.resize(_numLayers - 1);
        for (int i = 0; i < _numLayers - 1; i++)
            _weights[i].resize(_topology[i] + 1, _topology[i+1]);
        
    }
    
    
    
    void Network::setRandomWeights(){
        this->scaleWeightVectors();
        
        for (int i = 0; i < _numLayers - 1; i++)
            setRandomWeightsForLayer(i);
    }
    
    
    void Network::setRandomWeightsForLayer(unsigned int layer){
        _weights[layer] = Eigen::MatrixXd::Random(_topology[layer] + 1, _topology[layer + 1]);
    }
    
        
    void Network::setTopology(const std::vector<unsigned int>& topology){
        _topology = Eigen::toEigen(topology);
        initialize();
    }
    
    
    
    std::vector<std::vector<double>> Network::getWeightsAfterLayer(const unsigned int layer) const {
        
        Eigen::MatrixXd mat = getWeightsAfterLayerEigen(layer);
        size_t rows = mat.rows();

        std::vector<std::vector<double>> layerWeights;
        layerWeights.resize(rows);
        
        for (int i = 0; i < rows; i++)
            layerWeights[i] = Eigen::toStdVector(static_cast<Eigen::VectorXd>(mat.row(i)));
        
        return layerWeights;
    }
    
    
    
    Eigen::MatrixXd Network::getWeightsAfterLayerEigen(const unsigned int layer) const {
        
        // if provided layer number is outside the network topology
        // -1 is for there is no weight matrix after output layer
        if (layer >= numLayers() - 1)
            throw Exception("Network layer-number out of bound error");
        
        return _weights[layer];
    }
    
    
    void Network::setWeightsAfterLayer(const unsigned int layer,
                                       const std::vector<std::vector<double>>& weights){
        
        // if provided layer number is outside the network topology
        // -1 is for there is no weight matrix after output layer
        if (layer >= numLayers() - 1)
            throw Exception("Network layer-number out of bound error");
        
        // Check for possible dimension mismatch
        // The weight matrix between layer l1 and l2 is of dimension
        // (n_l1 + 1) x n_l2
        size_t numRows(weights.size()), numCols(weights[0].size());
        
        if (numRows != _topology[layer] + 1 ||
            numCols != _topology[layer + 1]) {
            std::stringstream temp;
            temp << "Here, I am expecting a " << _topology[layer] + 1 << " x " << _topology[layer + 1] << " matrix.";
            throw Exception({"Dimensions of provided weight matrix does not match the network topology.",
                "Note, W[layer] is a (nodes[layer] + 1) x nodes[layer + 1] matrix.",
                temp.str()});
        }
        // A possible new layer weights matrix
        Eigen::MatrixXd newWeights(numRows, numCols);
        
        // For each row
        for (int i = 0; i < numRows; i++)
            // All rows should have the same number of columns
            if (weights[i].size() != numCols)
                throw Exception("Provided weight matrix has variable length rows.");
            else
                newWeights.row(i) = Eigen::toEigen(weights[i]);
        // All's well change the weights
        _weights[layer] = newWeights;
        
    }
    
    
    void Network::printWeights() const {
        size_t size = _weights.size();
        
        for (int i = 0; i < size; i++) {
            std::cout << "Weights between (L:" << i << " , L:" << i+1 << "):" << std::endl;
            std::cout << _weights[i] << std::endl;
        }
    }

    
    void Network::setFunctionalityAtLayer(const unsigned int layer,
                                          FunctionalitySuite& suit) {
        
        // if provided layer number is outside the network topology
        if (layer >= numLayers())
            throw Exception("Network layer-number out of bound error");
        
        delete _functions[layer];
        _functions[layer] = &suit;
        
        // If the layer is output, set softmax flag off
        if (layer == _numLayers - 1)
            _isSoftmax = false;
    }
    
    
    void Network::setSoftmaxAtOutputLayer(){
        _isSoftmax = true;
        _functions[_numLayers - 1] = FunctionalitySuite::softmax();
        
    }
    
    
    
    void Network::processInput(const Eigen::Ref<const Eigen::VectorXd>& input,
                               std::vector<Eigen::VectorXd>& inputs,
                               std::vector<Eigen::VectorXd>& activations) {
        
        if (input.rows() != _numInputs)
            throw Exception("Input dimensions mis-match error for the input provided");
        
        // clear outputs and activations
        activations.clear();
        inputs.clear();
        
        // For input layer
        inputs.push_back(input);
        activations.push_back(operateFunction(_functions[0]->activationFunction(), input));
        
        // For subsequent layers
        for (int layer = 1; layer < _numLayers; layer++) {
            
            Eigen::VectorXd augmentedInput = Eigen::appendOnesAtHead(activations.back());
            
            inputs.push_back(_weights[layer - 1].transpose() * augmentedInput);
            if (layer == _numLayers - 1 && _isSoftmax) {
                // Softmax at the output
                Eigen::VectorXd output = operateFunction((double(*)(double))&std::exp, inputs.back());
                double sum = output.sum();
                activations.push_back(operateFunction([&](double x)->double { return x/sum; }, output));
            }
            else
                activations.push_back(operateFunction(_functions[layer]->activationFunction(),
                                                      inputs.back()));
            
        }
        
    }

    
    
    double Network::calculateLoss(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                  const double regularizer) {
        
        if (data.cols() != numOutputs() + numInputs())
            throw Exception({"Network architecture does not comply with the provided data.",
                "Note the data is assumed arranged in a matrix with one example (input, label) per row.",
                "The current network expects a row of size "+std::stringstream(static_cast<unsigned int>(numOutputs() +
                                                                                                         numInputs())).str()});
        
        // Number of features and labels
        size_t numFeatures(numInputs()), numClasses(numOutputs());
        
        
        // Number of examples
        size_t numExamples = data.rows();
        
        // Unregularized loss
        double unregularizedLoss(0.0);
        
        // for each example
        for (int example = 0; example < numExamples; example++) {
            
            std::vector<Eigen::VectorXd> inputs, activations;
            processInput(data.row(example).head(numFeatures), inputs, activations);
            
            // Update Loss
            unregularizedLoss += (operateFunction(_functions[_numLayers - 1]->lossFunction(),
                                                  activations.back(), data.row(example).tail(numClasses))
                                  ).sum();
        }
        
        // Regularization term :
        // (lambda/2)*[\sum (theta_ij)^2]   ... removing 0th row, corresponding to Bias
        double regularization(0.0);
        for (int i = 0; i < _numLayers - 1; i++)
            regularization += (Eigen::removeRow(_weights[i].cwiseProduct(_weights[i]), 0)).sum();
        
        // Regularized loss
        return ( (1.0 / numExamples) * unregularizedLoss + (regularizer / (2 * numExamples) ) * regularization);
    }
    
    double Network::calculateLoss(const Eigen::Ref<const Eigen::VectorXd>& input,
                                  const Eigen::Ref<const Eigen::VectorXd>& label,
                                  const double regularizer) {
        
        if (input.size() != numInputs() ||
            label.size() != numOutputs()) {
            throw Exception("Provided input or label dimensions do not comply with Network architecture.");
        }
        
        Eigen::MatrixXd example(1, numInputs() + numOutputs());
        example.row(0) << input.transpose() , label.transpose();
        
        return calculateLoss(example, regularizer);
    }

    
    
    double Network::train(const Eigen::MatrixXd& data, const double regularizer,
                          const bool verbose){
        
        // Assumption: data is arranged each row as an example
        // first m columns are features and next n columns labels
        
        // number of training iterations
        unsigned int numIterations(0);
        
        // Average loss per iteration
        double averageLoss(INFINITY);
        
        // Learning rate
        double eta(0.6);
        
        // Maximum adjustment applied to the weights
        double maxAdjustmentApplied(1.0);
        
        while (maxAdjustmentApplied > 1e-5
               && numIterations++ < 10000) {
            
            // Calculate loss and corrections to the weights
            std::vector<Eigen::MatrixXd> currentWeightCorrections;
            
            lossAndWeightCorrections(data, regularizer, averageLoss, currentWeightCorrections);
            
            // Adjust weights
            // Reset the maximum correction (%) applied to any weight coeff. in any layer
            maxAdjustmentApplied = 1.0;
            
            for (int layer = 0; layer < _numLayers - 1; layer++) {
                if (numIterations < 100 || numIterations % 100 == 0)
                    maxAdjustmentApplied = std::fabs(currentWeightCorrections[layer].cwiseQuotient(_weights[layer]).maxCoeff());
                // set new (corrected) weights
                setWeightsAfterLayer(layer,
                                     _weights[layer]
                                     - eta * currentWeightCorrections[layer]);
            }
            
            if (verbose && numIterations % 100 == 0)
                std::cout
                << "Train. Progress it #" << numIterations
                << ": Ave. loss = " << averageLoss
                << ", max. correction applied (%) = " << maxAdjustmentApplied*100
                << std::endl;
            
        }
        
        return averageLoss;
    }
    double Network::train(const char *dataFileName, const double regularizer, const bool verbose) {
        return train(Eigen::loadFromBinaryFile<double>(dataFileName), regularizer, verbose);
    }
    
    
    
    // HELPERS
    
    // A unit test method
    bool isValidNetworkImplementation(Network& testNet) {

        bool success = true;
        
        // Network topology
        size_t in(testNet.numInputs()),
        out(testNet.numOutputs()),
        numLayers(testNet.numLayers());
        
        // Generate a number of toy examples
        int numExamples(5);
        
        // Setup inputs and labels
        std::vector<Eigen::VectorXd> inputs, labels;
        for (int i=0; i < numExamples; i++) {
            inputs.push_back(Eigen::VectorXd::Random(in));
            Eigen::VectorXd label = operateFunction((double(*)(double))&std::fabs, Eigen::VectorXd::Random(out));
            double sum = label.sum();
            label /= sum;
            labels.push_back(label);
        }
        
        // correction matrices
        std::vector<Eigen::MatrixXd> numericalCorrections, gradientCorrections, networkWeights;
        for (int layer = 0; layer < numLayers - 1; layer++) {
            networkWeights.push_back(testNet.getWeightsAfterLayerEigen(layer));
            numericalCorrections.push_back(Eigen::MatrixXd::Zero(networkWeights.back().rows(), networkWeights.back().cols()));
            gradientCorrections.push_back(Eigen::MatrixXd::Zero(networkWeights.back().rows(), networkWeights.back().cols()));
        }
        
        // Perturbation precision term
        double epsilon = 1e-4;
        // for each example
        for (int example = 0; example < numExamples; example++) {
            
            // for each layer weights
            for (int layer = 0; layer < numLayers - 1; layer++) {
                
                size_t numRows(numericalCorrections[layer].rows()), numCols(numericalCorrections[layer].cols());
                
                // Perturbation matrix
                Eigen::MatrixXd perturb = Eigen::MatrixXd::Zero(numRows, numCols);
                // within this layer
                for (int row = 0; row < numRows; row++) {
                    for (int col = 0; col < numCols; col++) {
                        perturb(row, col) = epsilon;
                        testNet.setWeightsAfterLayer(layer, networkWeights[layer] + perturb);
                        double lossMin = testNet.calculateLoss(inputs[example], labels[example]);
                        testNet.setWeightsAfterLayer(layer, networkWeights[layer] - perturb);
                        double lossMax = testNet.calculateLoss(inputs[example], labels[example]);
                        numericalCorrections[layer](row, col) += ((lossMin - lossMax) / (2*epsilon));
                        perturb(row, col) = 0;
                    }
                }
                // reset weights on the network to original
                testNet.setWeightsAfterLayer(layer, networkWeights[layer]);
            }
            
            // for this example the correction by gradient method
            double loss;
            std::vector<Eigen::MatrixXd> corrections;
            testNet.unregularizedLossAndWeightCorrectionsForExample(inputs[example], labels[example],
                                                                    loss, corrections);
            for (int layer = 0; layer < numLayers - 1; layer++) {
                gradientCorrections[layer] += corrections[layer];
            }
            
        }
        
        // Check matching of correction matrices and output results
        std::string separator = "----------------------------------------------------------------\n";
        for (int layer = 0; layer < numLayers - 1; layer++) {
            gradientCorrections[layer] /= numExamples;
            numericalCorrections[layer] /= numExamples;
            double precision = 1e-9;
            bool layerCheck = false;
            while (!layerCheck && precision < 10*std::pow(epsilon, 2.0)) {
                layerCheck = numericalCorrections[layer].isApprox(gradientCorrections[layer], precision);
                precision *= 2;
            }
            // Even if one check fails, we have an unsuccessful test
            if (success && !layerCheck)
                success = false;
            
            std::cout << separator;
            std::cout << "Check for weight corrections between (L:" << layer << " ,L: " << layer + 1
            << ")" << (layerCheck ? " was successful!" : " failed!")
            << "\n" << "At Precision = " << precision << " for epsilon = " << epsilon <<  std::endl;
            std::cout << separator;
            if (!layerCheck) {
                std::cout << "I was expecting a correction matrix \n"
                << numericalCorrections[layer]
                << "\nBut the one I have from Gradient method is \n"
                << gradientCorrections[layer] << std::endl;
            }
        }
        
        return success;
    }
    
}