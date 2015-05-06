//
//  rntn.cpp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 12/14/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#include "rntn.hpp"

namespace NNetwork {
    
    void RNTN::init(){
        
        // Because Eigen uses srand
        srand(static_cast<unsigned int>( time(nullptr) ));
        
        // Default nonlinearity
        _nonlinearity = FunctionalitySuite::tanHyperbolic();
        
        // initialize Alphabet representation, L = [d x |Vocab|] matrix
        _wordRepMatrix = Eigen::random<Eigen::MatrixXd>(_dimWordRep, _vocabMap.size(), -0.0001, 0.0001);
        
        // initialize sentiment weight matrix, W_s = [numClasses x d] matrix
        _sentimentMatrix = Eigen::random<Eigen::MatrixXd>(_numClasses, _dimWordRep, -1.0, 1.0);
        
        // initialize weight matrix, W [ d x 2d ]
        _weightMatrix = Eigen::random<Eigen::MatrixXd>(_dimWordRep, 2*_dimWordRep, -1.0, 1.0);
        
        // initialize the interaction tensor
        if (_useTensor) {
            _interactionTensor.resize(_dimWordRep);
            for (auto& mat: _interactionTensor)
                mat = Eigen::random<Eigen::MatrixXd>(2 *_dimWordRep, 2 * _dimWordRep, -1.0, 1.0);
        }
    }
    
    
    
    bool RNTN::isLeaf(const NNetwork::PTree &p) const {
        
        // if p is null
        if (&p == nullptr)
            throw Exception("Can not confirm status as a 'leaf' of a NULL node.");
        
        return (p.left == nullptr && p.right == nullptr);
    }
    bool RNTN::isLeaf(const PTree_uptr& p) const {
        return isLeaf(*(p.get()));
    }
    
    
    bool RNTN::isRoot(const NNetwork::PTree &p) const {
        
        // if p is null
        if (&p == nullptr)
            throw Exception("Can not confirm status as 'root' of a NULL node.");
        
        return p.parent == nullptr;
        
    }
    bool RNTN::isRoot(const PTree_uptr& p) const {
        return isRoot(*(p.get()));
    }
    
    
    Eigen::VectorXd RNTN::getRepresentation(const PTree& p) {
        
        // If the node is not a leaf node
        if (!isLeaf(p))
            throw Exception("A non-leaf node is not a word.");
        
        // if the string exists in the vocabulary, return it
        // Else, insert it in the vocabulary, assign an arbitrary
        if (_vocabMap.count(p.phrase) == 0) {
            
            // insert it to the vocabulary
            size_t size = _vocabMap.size();
            _vocabMap[p.phrase] = size;
            // To the word representation matrix
            Eigen::appendColumnAtRight(_wordRepMatrix,
                                       Eigen::random<Eigen::VectorXd>(_dimWordRep, 1,
                                                                      -0.0001, 0.0001));
            
        }
        
        return static_cast<Eigen::VectorXd>(_wordRepMatrix.col(_vocabMap.at(p.phrase)));
        
    }
     Eigen::VectorXd RNTN::getRepresentation(const PTree_uptr& p) {
         return getRepresentation(*(p.get()));
     }
    
    
    // Sentiment Probability Vector = softmax(W_s x phraseRep)
    Eigen::VectorXd RNTN::evaluateSentiment(PTree& p) {
        
        // if provided tree is empty
        if (&p == nullptr)
            throw Exception("Can not evaluate sentiments for a NULL expression.");
            
        Eigen::VectorXd phraseRep;
        
        // if the provided node is a leaf node
        if (isLeaf(p))
            // pull the current representation from the representation matrix
            phraseRep = getRepresentation(p);
        
        // If not a leaf node, then
        else {
            // (a) the node should have an empty phrase string
            if (!p.phrase.empty())
                throw Exception("A non-leaf node with word string found.");
            
            // (b) the node should have an evaluated representation available
            if (p.phraseRep.rows() != _dimWordRep)
                evaluateRepresentation(p);
                
            phraseRep = p.phraseRep;
        }
        
        p.ratingEstimation = Eigen::operateFunction((double(*)(double))&std::exp,
                                                    _sentimentMatrix * phraseRep);
        
        // normalize
        double sum = p.ratingEstimation.sum();
        p.ratingEstimation /= sum;
        
        return p.ratingEstimation;
    
    }
    Eigen::VectorXd RNTN::evaluateSentiment(PTree_uptr& p) {
        return evaluateSentiment(*(p.get()));
    }
    
    
    
    
    Eigen::VectorXd RNTN::evaluateRepresentation(PTree& p){
        
        // If phrase is null valued
        if (&p == NULL)
            return Eigen::VectorXd::Zero(_dimWordRep);
        
        // If the phrase is a leaf
        if (isLeaf(p))
            return getRepresentation(p.phrase);
        
        // Else evaluate representations on subtrees
        auto leftPhraseRep = evaluateRepresentation(*(p.left));
        auto rightPhraseRep = evaluateRepresentation(*(p.right));
        
        // Append two children reps into a single vector of dim = 2d
        Eigen::VectorXd childrenRep(2 * _dimWordRep);
        childrenRep << leftPhraseRep, rightPhraseRep;
        
        // Tensor product, i'th component = [c1, c2] x V[i] x [c1, c2]^t
        Eigen::VectorXd tensorContribution = Eigen::VectorXd::Zero(_dimWordRep);
        if (_useTensor) {
            for (int i = 0; i < _dimWordRep; i++)
                tensorContribution.row(i) = childrenRep.transpose() * _interactionTensor[i] * childrenRep;
        }
        
        p.phraseRep = Eigen::operateFunction(_nonlinearity->activationFunction(),
                                             tensorContribution + _weightMatrix * childrenRep);
        
        return p.phraseRep;
    
    }
    Eigen::VectorXd RNTN::evaluateRepresentation(PTree_uptr& p){
        return evaluateRepresentation(*(p.get()));
    }
    
    
    double RNTN::loss(const Eigen::Ref<const Eigen::VectorXd>& output,
                      const Eigen::Ref<const Eigen::VectorXd>& label, const double regularizer){
        
        // cross-entropy loss function
        double loss = Eigen::operateFunction(FunctionalitySuite::softmax()->lossFunction(),
                                             output, label).sum();
        
        // regularization
        double regularization(0.0);
        if (regularizer) {
            // sentiment matrix
            regularization += _sentimentMatrix.cwiseProduct(_sentimentMatrix).sum();
            // weight matrix
            regularization += _weightMatrix.cwiseProduct(_weightMatrix).sum();
            // Representation matrix
            regularization += _wordRepMatrix.cwiseProduct(_wordRepMatrix).sum();
            // Interaction Tensor
            if (_useTensor)
                for (int i = 0; i < _dimWordRep; i++)
                    regularization += _interactionTensor[i].cwiseProduct(_interactionTensor[i]).sum();
        }
        
        // Regularized loss
        return loss + regularizer * regularization;
        
    }
    
    double RNTN::loss(PTree& p, const double regularizer) {
        
        // if p is empty
        if (&p == nullptr)
            return 0;
        
        // if p's sentiment was not evaluated
        if (p.ratingEstimation.rows() != _numClasses)
            evaluateSentiment(p);
        
        // Rating vector
        Eigen::VectorXd ratingVector = Eigen::VectorXd::Zero(_numClasses);
        ratingVector[p.rating - _lowerClassIndex] = 1;
        
        // Unregularized loss
        double lossVal = loss(p.ratingEstimation, ratingVector, 0.0);
        
        // Descend to subtrees (unregularized
        if (p.left)
            lossVal += loss(p.left, 0.0);
        if (p.right)
            lossVal += loss(p.right, 0.0);
        
        // Add regularization, if reached root
        double regularization(0.0);
        if (isRoot(p)) {
            if (regularizer) {
                // sentiment matrix
                regularization += _sentimentMatrix.cwiseProduct(_sentimentMatrix).sum();
                // weight matrix
                regularization += _weightMatrix.cwiseProduct(_weightMatrix).sum();
                // Representation matrix
                regularization += _wordRepMatrix.cwiseProduct(_wordRepMatrix).sum();
                // Interaction Tensor
                if (_useTensor)
                    for (int i = 0; i < _dimWordRep; i++)
                        regularization += _interactionTensor[i].cwiseProduct(_interactionTensor[i]).sum();
            }
        }
        
        return lossVal + regularizer * regularization;
    }
    
    double RNTN::loss(NNetwork::PTree_uptr& p, const double regularizer) {
        return loss(*(p.get()), regularizer);
    }
    
    double RNTN::loss(std::vector<PTree_uptr> &trees, const double regularizer) {
        double averageLoss(0.0);
        
        for (auto& p : trees) {
            averageLoss += loss(p, regularizer);
        }
        
        return averageLoss / trees.size();
    }
    
    
    void RNTN::correctionsToParamMatrices(NNetwork::PTree &p,
                                          Eigen::MatrixXd& correctionWsent,
                                          Eigen::MatrixXd& correctionW,
                                          std::vector<Eigen::MatrixXd>& correctionV,
                                          Eigen::VectorXd carryDown) {
        
        // if p is a NULL node
        if (&p == nullptr)
            return;
        
        // Within-bounds-check for p's rating
        size_t upperClassIndex = _lowerClassIndex + _numClasses - 1;
        if (p.rating < _lowerClassIndex ||
            p.rating > upperClassIndex) {
            std::string msg("Expected a class index between "
                            + std::to_string(_lowerClassIndex)
                            + " and "
                            + std::to_string(upperClassIndex));
            throw Exception({"Out-of-bound classification index at node.", msg});
        }
        
        // The Rating vector, t_i
        Eigen::VectorXd ratingVector = Eigen::VectorXd::Zero(_numClasses);
        ratingVector[p.rating - _lowerClassIndex] = 1;
        
            
        // if p is a root node, initialize correctionMatrices
        if (isRoot(p)) {
            // Sentiment Matrix
            correctionWsent = Eigen::MatrixXd::Zero(_numClasses, _dimWordRep);
            // W
            correctionW = Eigen::MatrixXd::Zero(_dimWordRep, 2*_dimWordRep);
            // V
            if (_useTensor) {
                correctionV.clear();
                for (int i = 0; i < _dimWordRep; i++)
                    correctionV.push_back( Eigen::MatrixXd::Zero(2 * _dimWordRep,
                                                                 2 * _dimWordRep) );
            }
            // Carry vector
            carryDown = Eigen::VectorXd::Zero(2* _dimWordRep);
        }
        
        
        // If p's sentiment etc was not already estimated
        if (p.ratingEstimation.rows() != _numClasses)
            evaluateSentiment(p);              // The phrase rep. will be automatically checked and calculated
    
        
        // Phrase representation, f(x_i)
        Eigen::VectorXd phraseRep = (isLeaf(p) ? getRepresentation(p) : p.phraseRep);
        
        
        // Correction to Ws
        correctionWsent += (p.ratingEstimation - ratingVector) * phraseRep.transpose();
        
        // Rest corrections need to be executed only if p is NOT a leaf
        if (isLeaf(p))
            return;
        
        // The softmax error at p, \delta^{p,s} [ y_i : ratingEstimation, f(x_i): phraseRep]
        Eigen::VectorXd softMaxError =
        (( _sentimentMatrix.transpose() * (p.ratingEstimation - ratingVector) )
        .cwiseProduct(Eigen::operateFunction(_nonlinearity->gradientFunction(),
                                             phraseRep) ) );
        
        // The complete error vector
        int isRightChild = (p.parent == nullptr) ? 0 : ( p.parent->left.get() == &p ? 0 : 1);
        auto completeErrVector = softMaxError + carryDown.segment(isRightChild * _dimWordRep, _dimWordRep);
        
        
        // Children vector
        Eigen::VectorXd childrenVector = Eigen::VectorXd::Zero(2 * _dimWordRep);
        if (p.left)
            childrenVector.head(_dimWordRep) = isLeaf(p.left) ? getRepresentation(*(p.left.get())) : p.left->phraseRep;
        if (p.right)
            childrenVector.tail(_dimWordRep) = isLeaf(p.right) ? getRepresentation(*(p.right.get())) : p.right->phraseRep;
        
        // Corrections to W
        correctionW += completeErrVector * childrenVector.transpose();
        
        // Corrections to Tensor, V
        if (_useTensor) {
            auto childVecMat = childrenVector * childrenVector.transpose();
            for (int i = 0; i < _dimWordRep; i++)
                correctionV[i] += completeErrVector[i] * childVecMat;
        }
        
        // Carry forward downword
        // Carry vector
        // W^t * \delta^{p, com}
        carryDown = _weightMatrix.transpose() * completeErrVector;
        // + S
        if (_useTensor) {
            for (int i = 0; i < _dimWordRep; i++)
                carryDown += completeErrVector[i] * ( _interactionTensor[i] + _interactionTensor[i].transpose() ) * childrenVector;
        }
        // Hadamard product
        carryDown = carryDown.cwiseProduct(Eigen::operateFunction(_nonlinearity->gradientFunction(),
                                                                  childrenVector));
        
        
        // To left subtree
        if (p.left)
            correctionsToParamMatrices(*(p.left.get()),
                                       correctionWsent, correctionW,
                                       correctionV, carryDown);
        // To right subtree
        if (p.right)
            correctionsToParamMatrices(*(p.right.get()),
                                       correctionWsent, correctionW,
                                       correctionV, carryDown);
        
    }
    
    
    double RNTN::train(const char *trainingFileName, const double regularizer, const bool verbose) {
     
        // Open the file
        std::ifstream trainFile;
        trainFile.open(trainingFileName);
        if (!trainFile.good() || trainFile.eof())
            throw Exception("Could not open training data file.");
        
        // number of training iterations
        unsigned int numIterations(0);
        
        // Learning rate
        double eta(0.2);
        
        // Maximum adjustment applied to the weights
        double maxAdjustmentApplied(1.0);
        
        // Average loss per iteration
        double averageLoss(0.0);
        
        while (maxAdjustmentApplied > 1e-5
               && numIterations++ < 1000) {
            
            // Calculate loss and corrections to the weights
            Eigen::MatrixXd correctionWs(Eigen::MatrixXd::Zero(_numClasses, _dimWordRep)),
            correctionW(Eigen::MatrixXd::Zero(_dimWordRep, 2*_dimWordRep));
            std::vector<Eigen::MatrixXd> correctionV;
            for (int slice = 0; slice < _dimWordRep; slice++)
                correctionV.push_back(Eigen::MatrixXd::Zero(2*_dimWordRep, 2*_dimWordRep));
            
            averageLoss = 0.0;
            
            size_t examples(0);
            
            while (trainFile.good() && !trainFile.eof()) {
                
                // Pull out an example
                std::string phrase;
                getline(trainFile, phrase, '\n');
                PTree_uptr phraseTree = NNetwork::buildTree(phrase);
                if (phraseTree == nullptr)
                    continue;
                
                examples++;
                
                // current loss
                averageLoss += loss(phraseTree, regularizer);
                
                // Current corrections
                Eigen::MatrixXd currCorrectionWs, currCorrectionW;
                std::vector<Eigen::MatrixXd> currCorrectionV;
                Eigen::VectorXd carry;

                correctionsToParamMatrices(*(phraseTree.get()),
                                           currCorrectionWs, currCorrectionW, currCorrectionV, carry);

                // Accumulate corrections
                correctionWs += currCorrectionWs;
                correctionW += currCorrectionW;
                for (int slice = 0; slice < _dimWordRep; slice++) {
                    correctionV[slice] += currCorrectionV[slice];
                }
            }
            
            // file ended, rewind for next iteration
            trainFile.clear();
            trainFile.seekg(0, std::ios::beg);
            
            // Average loss this iteration
            averageLoss /= examples;
            
            // Correct weights (and maximum applied correction
            // Ws
            _sentimentMatrix -= ((1.0 / examples) * eta * correctionWs +
                                 ( regularizer / (2 * examples) ) * _sentimentMatrix);
            
            maxAdjustmentApplied = std::fabs(correctionWs.cwiseQuotient(_sentimentMatrix).maxCoeff());
            
            // W
            _weightMatrix -= ((1.0 / examples) * eta * correctionW +
                              ( regularizer / (2 * examples) ) * _weightMatrix);
            
            maxAdjustmentApplied = std::max(maxAdjustmentApplied,
                                            correctionW.cwiseQuotient(_weightMatrix).maxCoeff());

            // V
            if (_useTensor) {
                for (int slice = 0; slice < _dimWordRep; slice++) {
                    _interactionTensor[slice] -= ((1.0 / examples) * eta * correctionV[slice] +
                                                  ( regularizer / (2 * examples) ) * _interactionTensor[slice]);
                    
                    maxAdjustmentApplied = std::max(maxAdjustmentApplied,
                                                    correctionV[slice].cwiseQuotient(_interactionTensor[slice]).maxCoeff());
                }
            }
            
            
            if (verbose /*&& numIterations % 100 == 0*/)
                std::cout << "Train. Progress it #" << numIterations
                << ": Ave. loss = " << averageLoss
                << ", max. correction applied (%) = " << maxAdjustmentApplied*100 << std::endl;
            
        }

        return averageLoss;
    }

    
    
    size_t RNTN::predict(NNetwork::PTree &p){
        
        // if p is empty
        if (&p == nullptr)
            throw Exception("Can not determine sentiment of a NULL expression.");
        
        // evaluate p's sentiment
        p.clearEstimations();
        
        evaluateSentiment(p);
        
        // Return the class with max probability
        size_t idxMax;
        p.ratingEstimation.maxCoeff(&idxMax);
        return idxMax + _lowerClassIndex;
    }
    size_t RNTN::predict(NNetwork::PTree_uptr& p) {
        return predict(*(p.get()));
    }
    
    
    // A unit test method
    bool isValidRntnImplementation(RNTN& testNet, const char* testFile,
                                   const size_t numExamples, const bool resetNet) {
       
        // Practicality checks
        if (numExamples > 20) {
            std::cerr << "WARNING : too many examples specified.\n"
            << "WARNING : Testing is computationally extensive and may require a lot of time to finish.\n"
            << "WARNING : Consider reducing number of examples (< 20) and rerunning the test.\n"
            << "WARNING : Test results do not depend significantly on this number."
            << std::endl;
        }
        
        size_t dimRep(testNet._dimWordRep),
        numClasses(testNet._numClasses);

        if (dimRep > 5 || numClasses > 5) {
            std::cerr << "WARNING : too large a test net specified.\n"
            << "WARNING : Testing is computationally extensive and may require a lot of time to finish.\n"
            << "WARNING : Consider using a lower dimensional representation (< 5) or number of classes (< 5) and rerunning the test.\n"
            << "WARNING : Test results do not depend significantly on this number."
            << std::endl;
        }

        // Open the testing cases files
        std::ifstream inFile;
        inFile.open(testFile);
        if (!inFile.good() || inFile.eof())
            throw Exception("Could not open test file.");
        
        // Pull in examples and Build vocabulary
        std::vector<NNetwork::PTree_uptr> phraseTrees;
        int example = 0;
        while (example++ < numExamples && inFile.good() && !inFile.eof()) {
            std::string test;
            std::getline(inFile, test, '\n');
            phraseTrees.push_back(NNetwork::buildTree(test));
            if (phraseTrees.back() == nullptr)
                phraseTrees.pop_back();
            else
                testNet.evaluateSentiment(phraseTrees.back());
        }
        // if not enough example fetched
        if (example != numExamples + 1) {
            std::string msg("User specified required of ");
            msg += std::to_string(numExamples);
            msg += " examples.";
            throw Exception({"The test file does not contain enough examples.", msg});
        }
        
        // corrections matrices
        // For storing Gradient method estimation:
        Eigen::MatrixXd sgCorrWs, sgCorrW;
        std::vector<Eigen::MatrixXd> sgCorrV;
        Eigen::VectorXd carry;
        
        // For numerical estiamtions
        Eigen::MatrixXd numCorrWs(Eigen::MatrixXd::Zero(numClasses, dimRep)),
        numCorrW(Eigen::MatrixXd::Zero(dimRep, 2* dimRep));
        std::vector<Eigen::MatrixXd> numCorrV(dimRep);
        for (int i = 0; i < dimRep; i++)
            numCorrV[i] = Eigen::MatrixXd::Zero(2*dimRep, 2*dimRep);
        
        // perturbation to be applied for numerical estimation
        double epsilon = 1e-4;
        
        
        // Estimate for each example
        for (int example = 0; example < numExamples; example++) {
            
            // corrections matrices
            Eigen::MatrixXd currentCorrWs, currentCorrW;
            std::vector<Eigen::MatrixXd> currentCorrV;
            Eigen::VectorXd carry;
            
            // For this example obtain corrections by gradient method
            try {
                if (example == 0) {
                    testNet.correctionsToParamMatrices(*(phraseTrees[example].get()),
                                                       sgCorrWs, sgCorrW, sgCorrV, carry);
                }
                else {
                    testNet.correctionsToParamMatrices(*(phraseTrees[example].get()),
                                                       currentCorrWs, currentCorrW, currentCorrV, carry);
                    sgCorrWs += currentCorrWs;
                    sgCorrW += currentCorrW;
                    for (size_t dim  = 0 ; dim < dimRep; dim++)
                        sgCorrV[dim] += currentCorrV[dim];
                }
            } catch (NNetwork::Exception& e) {
                std::cout << e.what() << std::endl;
            }
            
            // Clear the estimations
            phraseTrees[example]->clearEstimations();
            
            // Corrections by numerical estimation
            
            // For Ws
            for (int row = 0; row < numClasses; row++) {
                for (int col = 0; col < dimRep; col++) {
                    
                    testNet._sentimentMatrix(row, col) += epsilon;
                    
                    phraseTrees[example]->clearEstimations();
                    double lossPlus = testNet.loss(phraseTrees[example], 0);
                    
                    testNet._sentimentMatrix(row, col) -= 2*epsilon;    // 2 for first resetting to original value
                    phraseTrees[example]->clearEstimations();
                    double lossMin = testNet.loss(phraseTrees[example], 0);
                    
                    numCorrWs(row, col) += ( (lossPlus - lossMin) / (2*epsilon) );
                    
                    phraseTrees[example]->clearEstimations();
                    testNet._sentimentMatrix(row, col) += epsilon;     // reset to original value
                    
                }
            }
            
            
            // For W
            for (int row = 0; row < dimRep; row++) {
                for (int col = 0; col < 2*dimRep; col++) {
                    testNet._weightMatrix(row, col) += epsilon;
                    phraseTrees[example]->clearEstimations();
                    double lossPlus = testNet.loss(phraseTrees[example], 0);
                    
                    testNet._weightMatrix(row, col) -= 2*epsilon;    // 2 for first resetting to original value
                    phraseTrees[example]->clearEstimations();
                    double lossMin = testNet.loss(phraseTrees[example], 0);
                    
                    numCorrW(row, col) += ( (lossPlus - lossMin) / (2*epsilon) );
                    
                    phraseTrees[example]->clearEstimations();
                    testNet._weightMatrix(row, col) += epsilon;     // reset to original value
                    
                }
            }
            
            // For V
            for (int slice = 0; slice < dimRep; slice++) {
                for (int row = 0; row < 2*dimRep; row++) {
                    for (int col = 0; col < 2*dimRep; col++) {
                        testNet._interactionTensor[slice](row, col) += epsilon;
                        phraseTrees[example]->clearEstimations();
                        double lossPlus = testNet.loss(phraseTrees[example], 0);
                        
                        testNet._interactionTensor[slice](row, col) -= 2*epsilon;    // 2 for first resetting to original value
                        phraseTrees[example]->clearEstimations();
                        double lossMin = testNet.loss(phraseTrees[example], 0);
                        
                        numCorrV[slice](row, col) += ( (lossPlus - lossMin) / (2*epsilon) );
                        
                        phraseTrees[example]->clearEstimations();
                        testNet._interactionTensor[slice](row, col) += epsilon;     // reset to original value
                        
                    }
                }
            }
        }
        
        // Confirm consistancy of Ws
        double precision = std::pow(epsilon, 2.0);
        bool success = false;
        
        while (!success && precision < 50*std::pow(epsilon, 2.0)) {
            success = numCorrWs.isApprox(sgCorrWs, precision);
            precision *= 2;
        }
        
        if (!success) {
            std::cerr << "Test failed for Ws" << std::endl;
            std::cerr << "Expected:\n" << numCorrWs << "\nByGradient:\n" << sgCorrWs << std::endl;
            return false;
        }
        
        // Average
        sgCorrW /= numExamples;
        numCorrW /= numExamples;
        
        // Confirm consistancy of W
        precision = 1e-4;
        success = false;
        
        while (!success && precision < 0.01) {
            success = numCorrW.isApprox(sgCorrW, precision);
            precision *= 2;
        }
        
        if (!success) {
            std::cerr << "Test failed for W" << std::endl;
            std::cerr << "Expected:\n" << numCorrW << "\nByGradient:\n" << sgCorrW << std::endl;
            return false;
        }
        
        // Consistency of V slices
        for (int slice = 0; slice < dimRep; slice++) {
            sgCorrV[slice] /= numExamples;
            numCorrV[slice] /= numExamples;
            
            precision = 1e-4;
            success = false;
            
            while (!success && precision < 0.01) {
                success = numCorrV[slice].isApprox(sgCorrV[slice], precision);
                precision *= 2;
            }
            
            if (!success) {
                std::cerr << "Test failed for V[" << slice << "]" << std::endl;
                std::cerr << "Expected:\n" << numCorrV[slice] << "\nByGradient:\n" << sgCorrV[slice] << std::endl;
                return false;
            }
            
        }
        
        // Clear the vocabulary of RNTN built up during testing
        if (resetNet)
            testNet._vocabMap.clear();
        
        return success;
    }

    
    
}