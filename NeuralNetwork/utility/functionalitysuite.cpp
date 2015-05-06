//
//  functionalitySuit.cpp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 11/26/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#include "functionalitysuite.hpp"

FunctionalitySuite::~FunctionalitySuite(){
    
}


FunctionalitySuite* FunctionalitySuite::logistic(){
    return new Logistic;
}

FunctionalitySuite* FunctionalitySuite::tanHyperbolic() {
    return new TanHyperbolic;
}

FunctionalitySuite* FunctionalitySuite::softmax() {
    return new SoftMax;
}

FunctionalitySuite* FunctionalitySuite::customSuite(const std::function<double (double)> &activationFunction,
                                                    const std::function<double (double)> &gradientFunction,
                                                    const std::function<double (double, double)> &lossFunction) {
    return new CustomSuite(activationFunction,
                           gradientFunction,
                           lossFunction);
}