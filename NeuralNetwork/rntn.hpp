//
//  rntn.h
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 12/14/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#ifndef __NeuralNetwork__rntn__
#define __NeuralNetwork__rntn__

#include <unordered_map>
#include <string>
#include <ctime>

#include <Eigen/Dense>

#include "utility-rntn.hpp"
#include "functionalitysuite.hpp"

namespace NNetwork {
    
    typedef std::unordered_map<std::string, unsigned long> Vocabulary;
    
    class RNTN {
    public:
        
        // constructor
        RNTN(const size_t dimWordRep,
             const size_t numClasses,
             const size_t lowerClassIndex,
             const bool useTensor = true)
        : _dimWordRep(dimWordRep),
        _numClasses(numClasses),
        _lowerClassIndex(lowerClassIndex),
        _useTensor(useTensor) {
            init();
        }
        
        // Set vocabulary
        void setVocabulary(Vocabulary& vocab)                     { _vocabMap = vocab; }
        // Get vocabulary
        Vocabulary getVocabulary(void) const                      { return _vocabMap; }
        // Set non-linearity
        void setNonlinearlity(const FunctionalitySuite& nonlinearity);
        
        // Train the network
        double train(const char* trainingFile, const double regularizer = 0,
                     const bool verbos = true);
        // Predict sentiment
        size_t predict(PTree& p);
        size_t predict(PTree_uptr& p);

        
        // A unit_test method
        friend bool isValidRntnImplementation(RNTN& testNet, const char* testFile,
                                              const size_t numExamples, const bool resetNet);
        
        
    private:
        
        size_t _dimWordRep, _lowerClassIndex, _numClasses;
        Eigen::MatrixXd _sentimentMatrix, _wordRepMatrix, _weightMatrix;
        std::vector<Eigen::MatrixXd> _interactionTensor;
        Vocabulary _vocabMap;
        FunctionalitySuite* _nonlinearity;
        bool _useTensor;
        
        // No copying of RNTN network
        RNTN& operator=(const RNTN& o) {
            return *this;
        }
        // No assignments
        RNTN(const RNTN& o)
        : _vocabMap(o._vocabMap) {
            
        }
        
        // Initialize network to random state
        void init(void);
        // Is the given node a leaf
        bool isLeaf(const PTree& p) const;
        bool isLeaf(const PTree_uptr& p) const;
        // Is a given node a root
        bool isRoot(const PTree& p) const;
        bool isRoot(const PTree_uptr& p) const;
        // Get current representation of a word at the (leaf) node
        Eigen::VectorXd getRepresentation(const PTree& p);
        Eigen::VectorXd getRepresentation(const PTree_uptr& p);
        // Evaluate the sentiment for the given phrase at a given node.
        Eigen::VectorXd evaluateSentiment(PTree& p);
        Eigen::VectorXd evaluateSentiment(PTree_uptr& p);
        // Evaluate representation for the tree rooted at given node
        Eigen::VectorXd evaluateRepresentation(PTree& p);
        Eigen::VectorXd evaluateRepresentation(PTree_uptr& p);
        // Loss function (softmax)
        double loss(const Eigen::Ref<const Eigen::VectorXd>& output,
                    const Eigen::Ref<const Eigen::VectorXd>& label, const double regularizer = 0);
        double loss(PTree& p, const double regularizer = 0);
        double loss(PTree_uptr& p, const double regularizer = 0);
        double loss(std::vector<PTree_uptr>& trees, const double regularizer = 0);
        // Corrections to Ws, W, V
        void correctionsToParamMatrices(PTree& p,
                                        Eigen::MatrixXd& correctionWsent,
                                        Eigen::MatrixXd& correctionW,
                                        std::vector<Eigen::MatrixXd>& correctionV,
                                        Eigen::VectorXd carryDown);
        void correctionsToParamMatrices(PTree_uptr& p,
                                        Eigen::MatrixXd& correctionWsent,
                                        Eigen::MatrixXd& correctionW,
                                        std::vector<Eigen::MatrixXd>& correctionV,
                                        Eigen::VectorXd carryDown);
        void correctionsToParamMatrices(std::vector<PTree_uptr>& trees,
                                        Eigen::MatrixXd& correctionWsent,
                                        Eigen::MatrixXd& correctionW,
                                        std::vector<Eigen::MatrixXd>& correctionV,
                                        Eigen::VectorXd carryDown);
        
    };
    
    
    // Implementation test method
    bool isValidRntnImplementation(RNTN& testNet, const char* testFile,
                                   const size_t numExamples, const bool resetNet = true);
    
}

#endif /* defined(__NeuralNetwork__rntn__) */
