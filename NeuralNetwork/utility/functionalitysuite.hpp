//
//  functionalitySuit.h
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 11/26/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#ifndef __NeuralNetwork__functionalitySuite__
#define __NeuralNetwork__functionalitySuite__

#include <iostream>
#include <cmath>

#include "functionalitysuite.hpp"

class FunctionalitySuite {
public:
    // Abstract class
    virtual ~FunctionalitySuite() = 0;
    
    // member functions
    std::function<double(double)> activationFunction(void)              { return _activationFunction; }
    std::function<double(double)> gradientFunction(void)                { return _gradientFunction; }
    std::function<double(double, double)> lossFunction(void)            { return _lossFunction; }
    
    // Known functions
    static FunctionalitySuite* logistic(void);
    static FunctionalitySuite* tanHyperbolic(void);
    static FunctionalitySuite* softmax(void);
    static FunctionalitySuite* customSuite(const std::function<double (double)>& activationFunction,
                                           const std::function<double (double)>& gradientFunction,
                                           const std::function<double (double, double)>& lossFunction);
    
    // Setters
    void setActivationFunction(std::function<double(double)> activiityFunction) {
        _activationFunction = activiityFunction;
    }
    
    void setGradientFunction(std::function<double(double)> gradientFunction) {
        _gradientFunction = gradientFunction;
    }
    
    void setLossFunction(std::function<double(double, double)> lossFunction) {
        _lossFunction = lossFunction;
    }

    
protected:
    std::function<double(double)> _activationFunction, _gradientFunction;
    std::function<double(double, double)> _lossFunction;
    
    
};


class Logistic: public FunctionalitySuite {
public:
    Logistic(){
        _activationFunction = [](double x)->double
        {
            return 1.0 / (1 + std::exp(-x));
        };
        _gradientFunction = [](double activation)->double { return activation * ( 1 - activation ); };
        _lossFunction = [](double output, double label)->double
        {
            // To avoid 0*log0
            if (output == 0 || output == 1)
                return (output == label) ? 0 : INFINITY;
            
            return - (label * std::log(output) +
                      (1 - label) * std::log(1 - output));
        };

    }
    
};



class TanHyperbolic: public FunctionalitySuite {
public:
    TanHyperbolic(){
        _activationFunction = (double(*)(double))&std::tanh;
        _gradientFunction = [](double activation)->double {
            // grad( tanh(x) ) = sech^2(x) = 1 - tanh^2(x) = 1 - activation
            return 1.0 - std::pow(activation, 2.0);
        };
        _lossFunction = [](double output, double label)->double {
            
            // To avoid 0*log0
            if (std::fabs(output) == 1)
                return (output == label) ? -0.5 * (2*std::log(2)) : INFINITY;
            
            return (-0.5)*((1 - label) * std::log(std::fabs(1 - output)) +
                           (1 + label) * std::log(std::fabs(1 + output)));
        };
    }
    
};



class SoftMax: public FunctionalitySuite {
public:
    SoftMax(){
        _activationFunction = [](double x)->double { return 0; };
        _gradientFunction = [](double activation)->double { return activation * (1 - activation); };
        _lossFunction = [&](double output, double label)->double {
            
            // 0*log0 = 0
            if (output == 0)
                return (output == label) ? 0 : INFINITY;
            
            return - label * std::log(output);
        };
    }
};



class CustomSuite: public FunctionalitySuite {
public:
    CustomSuite(const std::function<double (double)>& activationFunction,
                const std::function<double (double)>& gradientFunction,
                const std::function<double (double, double)>& lossFunction) {
        _activationFunction = activationFunction;
        _gradientFunction = gradientFunction;
        _lossFunction = lossFunction;
    }
};

#endif /* defined(__NeuralNetwork__functionalitySuite__) */
