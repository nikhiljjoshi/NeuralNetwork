//
//  network.ipp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 12/5/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//


// The public methods that take Eigen::Ref as input argument
// since Eigen::Ref is (Expression) templatized
// calling it requires definitions avaiable


void Network::setWeightsAfterLayer(const unsigned int layer,
                                   const Eigen::Ref<const Eigen::MatrixXd>& weights){

    // if provided layer number is outside the network topology
    // -1 is for there is no weight matrix after output layer
    if (layer >= numLayers() - 1)
        throw Exception("Network layer-number out of bound error");
    
    // Check for possible dimension mismatch
    // The weight matrix between layer l1 and l2 is of dimension
    // (n_l1 + 1) x n_l2
    if (weights.rows() != _topology[layer] + 1 ||
        weights.cols() != _topology[layer + 1])
        throw Exception("Provided weight matrix dimension does not match with network topology");
    
    // All's well change the weights
    _weights[layer] = weights;
}



void Network::unregularizedLossAndWeightCorrectionsForExample(const Eigen::Ref<const Eigen::VectorXd>& input,
                                                              const Eigen::Ref<const Eigen::VectorXd>& label,
                                                              double& loss,
                                                              std::vector<Eigen::MatrixXd>& weightCorrections) {
    
    // Check for Label - output dimension match
    // (input will be matched in the processInput method below)
    if (label.rows() != _numOutputs)
        throw Exception("Network output-label dimensions mis-match error.");
    
    // Produced output for the current input
    std::vector<Eigen::VectorXd> inputs, activations;
    processInput(input, inputs, activations);
    
    // The loss
    loss = (operateFunction(_functions[_numLayers - 1]->lossFunction(),
                            activations.back(), label)
            ).sum();
    
    // The correction vectors
    std::vector<Eigen::VectorXd> delta(_numLayers);
    
    // From output towards input
    for (size_t layer = _numLayers - 1; layer > 0; layer--) {
        
        // For output layer
        if (layer == _numLayers - 1)
            delta[layer] = activations.back() - label;
        else {// for all other layers
            delta[layer] = Eigen::removeRow((_weights[layer] *
                                             delta[layer + 1]).cwiseProduct(operateFunction(_functions[layer]->gradientFunction(),
                                                                                            Eigen::appendOnesAtHead(activations[layer]))), 0);
        }
        
    }
    
    weightCorrections.clear();
    weightCorrections.resize(_numLayers - 1);
    
    for (size_t layer = 0; layer < _numLayers - 1; layer++)
        weightCorrections[layer] = Eigen::appendOnesAtHead(activations[layer]) * delta[layer + 1].transpose();
    
}



void Network::lossAndWeightCorrections(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                       const double regularizer,
                                       double& loss,
                                       std::vector<Eigen::MatrixXd>& weightCorrections) {
    
    if (data.cols() != numOutputs() + numInputs())
        throw Exception({"Network architecture does not comply with the provided data.",
            "Note the data is assumed arranged one example in each row.",
            "The current network expects a row of size "+std::stringstream(static_cast<unsigned int>(numOutputs() +
                                                                                                     numInputs())).str()});
    
    // Number of features and labels
    size_t numFeatures(numInputs()), numClasses(numOutputs());
    
    // Number of examples
    size_t numExamples = data.rows();
    
    if (numExamples < 1) {
        std::cout << "** Warning: No training examples provided.\n"
        << "** Warning: Loss and corrections not calculated." << std::endl;
        return;
    }
    
    
    // Initialized corrections with the first example output
    unregularizedLossAndWeightCorrectionsForExample(data.row(0).head(numFeatures),
                                                    data.row(0).tail(numClasses),
                                                    loss,
                                                    weightCorrections);
    
    
    // Update with all rest examples
    for (int example = 1; example < numExamples; example++) {
        double thisExampleLoss(0.0);
        std::vector<Eigen::MatrixXd> thisExampleCorrections;
        
        unregularizedLossAndWeightCorrectionsForExample(data.row(example).head(numFeatures),
                                                        data.row(example).tail(numClasses),
                                                        thisExampleLoss, thisExampleCorrections);
        
        // update loss
        loss += thisExampleLoss;
        
        // Update weight corrections for each layer
        for (size_t layer = 0; layer < _numLayers - 1; layer++)
            weightCorrections[layer] += thisExampleCorrections[layer];
        
    }
    
    // Add regularization
    // for loss
    double regularization(0.0);
    for (int i = 0; i < _numLayers - 1; i++)
        regularization += (Eigen::removeRow(_weights[i].cwiseProduct(_weights[i]), 0)).sum();
    
    
    loss = loss / numExamples + (regularizer / (2 * numExamples)) * regularization;
    
    // for weight corrections
    for (size_t layer = 0; layer < _numLayers - 1; layer++) {
        weightCorrections[layer] =  ((1.0 / numExamples) * weightCorrections[layer] +
                                     (regularizer / numExamples) * Eigen::resetRowToZero(_weights[layer], 0));
    }
    
}
