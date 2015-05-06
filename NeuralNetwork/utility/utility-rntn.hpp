//
//  utility-rntn.hpp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 12/17/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#ifndef NeuralNetwork_utility_rntn_hpp
#define NeuralNetwork_utility_rntn_hpp

#include <cmath>

#include "utility.hpp"

namespace NNetwork {
    
    struct PTree;
    typedef std::unique_ptr<PTree> PTree_uptr;
    
    // A Phrase tree
    struct PTree {
        PTree_uptr left, right;
        PTree* parent;
        std::string phrase;
        double rating;
        Eigen::VectorXd phraseRep, ratingEstimation;
        
        PTree(std::string p)
        : phrase(p) {
            parent = nullptr;
            rating = -INFINITY;
            phraseRep = Eigen::VectorXd::Zero(0);
            ratingEstimation = Eigen::VectorXd::Zero(0);
        }
        
        void clearEstimations(void) {
            phraseRep = Eigen::VectorXd::Zero(0);
            ratingEstimation = Eigen::VectorXd::Zero(0);
            
            // clear those of the children
            if (left)
                left->clearEstimations();
            if (right)
                right->clearEstimations();
        }
    };
    
    
    // Traverse the string without counting whitespace
    inline void nextNonWsChar(std::string& str, unsigned int& i) {
        
        if (i >= str.size())
            return;
        
        do {
            i++;
        } while (str[i] == ' ' && i < str.size());
        
    }
    
    
    // Build tree from a given string
    inline PTree_uptr buildTreeHelper(std::string& str, unsigned int& position) {
        
        if (str.empty())
            return nullptr;
        
        
        if (str[position] != '(')
            throw Exception("String format doesn't comply with PTB format.");
        
        // remove leading '('
        nextNonWsChar(str, position);
        
        // New empty node
        PTree_uptr node = std::make_unique<PTree>("");
        
        if (str[position] >= '0' && str[position] <= '5') {
            node->rating = str[position] - '0';
            nextNonWsChar(str, position);
        }
        
        while (str[position] != '(' && str[position] != ')') {
            node->phrase.push_back(str[position]);
            nextNonWsChar(str, position);
        }
        
        if (str[position] == ')') {
            nextNonWsChar(str, position);
            return node;
        }
        
        if (str[position] == '(') {
            node->left= buildTreeHelper(str, position);
            node->left->parent = node.get();
        }
        
        if (str[position] == ')') {
            nextNonWsChar(str, position);
            return node;
        }
        
        if (str[position] == '(') {
            node->right = buildTreeHelper(str, position);
            node->right->parent = node.get();
        }
        
        if (str[position] != ')')
            throw Exception("String does not comply with the PTB format.");
        
        // remove trailing ')'
        nextNonWsChar(str, position);
        
        return node;
    }
    
    
    inline PTree_uptr buildTree(std::string& str) {
        
        unsigned int pos(0);
        return buildTreeHelper(str, pos);
    }
    
}  // namespace NNetwork

#endif
