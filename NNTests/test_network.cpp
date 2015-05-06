//
//  test_network.cpp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 11/15/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#include "network.hpp"

#define BOOST_TEST_MODULE test_network
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(tests_network)


BOOST_AUTO_TEST_CASE(topologyTests) {
    
    unsigned int input(10), output(12);
    
    NNetwork::Network n1({input, output});
    
    BOOST_CHECK_EQUAL(n1.numInputs(), input);
    BOOST_CHECK_EQUAL(n1.numOutputs(), output);
    BOOST_CHECK_EQUAL(n1.numLayers(), 2);
}



BOOST_AUTO_TEST_CASE(networkImplementationTests) {

    // A small test network
    unsigned int in(4), hidden(5), hidden2(4), out(2);
    NNetwork::Network testNet({in, hidden, hidden2, out});
    
    // A pure logistic node network
    std::cout << "Logistic Network test started!" << std::endl;
    BOOST_ASSERT(NNetwork::isValidNetworkImplementation(testNet));
    std::cout << "Logistic Network test ended!" << std::endl;

    
    // A logistic network with output softmax
    testNet.setSoftmaxAtOutputLayer();
    std::cout << "Logistic Network with softmax output test started!" << std::endl;
    BOOST_ASSERT(NNetwork::isValidNetworkImplementation(testNet));
    std::cout << "Logistic Network with softmax output test ended!" << std::endl;
    
    // A tan hyperbolic network
    size_t numLayers = testNet.numLayers();
    for (int layer = 0; layer < numLayers; layer++)
        testNet.setFunctionalityAtLayer(layer, *FunctionalitySuite::tanHyperbolic());

    std::cout << "TanHyperbolic Network with softmax output test started!" << std::endl;
    BOOST_ASSERT(NNetwork::isValidNetworkImplementation(testNet));
    std::cout << "TanHyperbolic Network with softmax output test ended!" << std::endl;
     
}


BOOST_AUTO_TEST_CASE(networkTrainingTest) {
// 
//    // Just in case
//    srand(static_cast<unsigned int>(time(NULL)));
//    
//    // A test network
//    NNetwork::Network testNet({2, 1});
//    
//    // Network topology
//    size_t in(testNet.numInputs()),
//    out(testNet.numOutputs());
//    
//    // Generate a number of toy examples
//    int numExamples(50);
//    
//    // Original weights (some known random values)
//    auto originalWeights = Eigen::VectorXd(in+out);
//    originalWeights << 5.5, 0.3, 1.4;
//    
//    // Setup inputs and labels
//    Eigen::MatrixXd data(numExamples, in + out);
//    // TEMP (passing eigen doesn't work :()
//    std::vector<std::vector<double>> stdData(numExamples);
//    for (int i=0; i < numExamples; i++) {
//        auto inputs = Eigen::random<Eigen::VectorXd>(in, 1, -10, 10);
//        
//        Eigen::VectorXd inAug(in+1);
//        inAug << Eigen::appendOnesAtHead(inputs);
//        
//        // prepare the label
//        auto labels = 1.0 / (1 + std::exp(-(static_cast<double>(originalWeights.transpose()*inAug))));
//        
//        Eigen::VectorXd example(in + out);
//        example << inputs, labels;
//        data.row(i) = example;
//        stdData[i] = Eigen::toStdVector(example);
//        std::cout << example.transpose() << std::endl;
//    }
//    
//    double tolerance(0.01);
//    double regularizer(0.01);
//    
//    
//    testNet.train(stdData, regularizer, tolerance);
//    
//    testNet.printWeights();
//
    
    
}


BOOST_AUTO_TEST_SUITE_END()