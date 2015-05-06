//
//  neuralnetwork.h
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 10/15/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#ifndef __NeuralNetwork__neuralnetwork__
#define __NeuralNetwork__neuralnetwork__

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Dense>

#include <iostream>
#include <ctime>
#include <vector>
#include <initializer_list>

#include "utility/utility.hpp"
#include "utility/functionalitysuite.hpp"

namespace NNetwork {
    
    class Network {
    public:
        // constructor
        Network(std::vector<unsigned int>&& topology) {
            _topology =  Eigen::toEigen(topology);
            initialize();
        }
        
        Network(std::initializer_list<unsigned int>& ls)
        : Network(std::vector<unsigned int>(ls))
        {
        }
        
        // Destructor
        ~Network(){
            for (int i = 0; i < _numLayers; i++)
                delete _functions[i];
        }
        
        
        // member functions
        // Setters/Getters
        // network topology
        void setTopology(const std::vector<unsigned int>& topology);
        void setTopology(const std::initializer_list<unsigned int>& topology);
        std::vector<unsigned int> getTopology(void)  const      { return Eigen::toStdVector(_topology); }
        size_t numInputs(void)   const                          { return _numInputs; }
        size_t numOutputs(void)  const                          { return _numOutputs; }
        size_t numLayers(void)   const                          { return _numLayers; }
        // weights
        std::vector<std::vector<double>> getWeightsAfterLayer(const unsigned int layer) const;
        Eigen::MatrixXd getWeightsAfterLayerEigen(const unsigned int layer) const;
        void printWeights(void) const;
        void setRandomWeights(void);
        void setRandomWeightsForLayer(unsigned int layer);
        void setWeightsAfterLayer(const unsigned int layer, const std::vector<std::vector<double>>& weights);
        void setWeightsAfterLayer(const unsigned int layer, const Eigen::Ref<const Eigen::MatrixXd>& weights);

        // activation function
        void setFunctionalityAtLayer(const unsigned int layer, FunctionalitySuite& suit);
        // make it a softmax output
        void setSoftmaxAtOutputLayer(void);
        bool isSoftmaxAtOutputLayer(void)                      { return _isSoftmax; }
        
        
        // Train the network on given examples
        // Data is assumed given in standard format :
        // One row = one (training) example n input/feature columns followed by m output/label columns
        double train(const char* dataFileName, const double regularizer,
                     const bool verbose = true);
        double train(const Eigen::MatrixXd& data, const double regularizer,
                     const bool verbose = true);
        
        // Predict
        unsigned int predict(const std::vector<double>& input);
        
        
        // Test methods
        friend bool isValidNetworkImplementation(Network& testNet);
 
        
    private:

        Eigen::VectorXu _topology;
        std::vector<Eigen::MatrixXd> _weights;
        std::vector<FunctionalitySuite*> _functions;
        size_t _numInputs, _numOutputs, _numLayers;
        bool _isSoftmax;
        
        // Member functions
        // initializers
        void initialize(void);
        void scaleWeightVectors(void);
        void setupDefaultNetwork(void);
        
        // Process input to produce output
        void processInput(const Eigen::Ref<const Eigen::VectorXd>& input,
                          std::vector<Eigen::VectorXd>& inputs,
                          std::vector<Eigen::VectorXd>& activations);
        // Calculate loss for the given input - label pair
        double calculateLoss(const Eigen::Ref<const Eigen::MatrixXd>& data,
                             const double regularizer = 0);
        double calculateLoss(const Eigen::Ref<const Eigen::VectorXd>& input,
                             const Eigen::Ref<const Eigen::VectorXd>& label,
                             const double regularizer = 0);
        
        // Calculate Error Delta by back Propagation
        void unregularizedLossAndWeightCorrectionsForExample(const Eigen::Ref<const Eigen::VectorXd>& input,
                                                             const Eigen::Ref<const Eigen::VectorXd>& label,
                                                             double& loss,
                                                             std::vector<Eigen::MatrixXd>& weightCorrections);
        void lossAndWeightCorrections(const Eigen::Ref<const Eigen::MatrixXd>& data,
                                      const double regularizer,
                                      double& loss,
                                      std::vector<Eigen::MatrixXd>& weightCorrections);
    };
    

    // Network implementation checks
    bool isValidNetworkImplementation(Network& testNet);
    
    
    // include implementation file
#include "network.ipp"
    
}


#endif /* defined(__NeuralNetwork__neuralnetwork__) */
