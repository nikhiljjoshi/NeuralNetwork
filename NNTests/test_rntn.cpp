//
//  test_rntn.cpp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 12/22/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#include "rntn.hpp"

#define BOOST_TEST_MODULE test_network
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(rntn_tests)

BOOST_AUTO_TEST_CASE(networkImplementationTests) {
    
    // A small test network
    size_t dimRepresentation(3), numClasses(5), lowerClassIndex(0);
    NNetwork::RNTN testNet(dimRepresentation, numClasses, lowerClassIndex);

    // A test file
    std::string testFile("data/trees/dev.txt");
    size_t numExamples(10);
    
    std::cout << "RNTN test started!" << std::endl;
    BOOST_ASSERT(NNetwork::isValidRntnImplementation(testNet, testFile, numExamples));
    std::cout << "RNTN test ended!" << std::endl;
}


BOOST_AUTO_TEST_SUIT_END()