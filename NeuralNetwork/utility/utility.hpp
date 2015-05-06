//
//  utility.hpp
//  NeuralNetwork
//
//  Created by Nikhil Joshi on 10/15/14.
//  Copyright (c) 2014 Nikhil Joshi. All rights reserved.
//

#ifndef NeuralNetwork_utility_hpp
#define NeuralNetwork_utility_hpp


#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <vector>


// NNetwork exception
namespace NNetwork {
    
    class Exception: public std::exception {
    public:
        Exception(const std::string& message)
        : _message(message){
            _message = "NNException:: " + _message;
            
        }
        Exception(std::initializer_list<std::string> messages) {

            std::stringstream msgStream;
            for (auto& message: messages)
                msgStream << "NNException:: " << message << "\n";
            
            _message = msgStream.str();
            
        }
        
        const char* what(void){
            std::cerr << "NNException occurred!" << std::endl;
            return _message.c_str();
        }
        
    private:
        std::string _caller, _message;
    };
    
}


// New typedefs, utilities for Eigen
namespace Eigen {
    typedef Matrix<unsigned int, Dynamic, 1> VectorXu;
    
    
    // Convert std::vector to Eigen Matrix
    template <typename T>
    inline Eigen::Matrix<T, Dynamic, 1> toEigen(const std::vector<T>& stdVec ) {
        return Eigen::Matrix<T, Dynamic, 1>::Map(stdVec.data(), stdVec.size());
    }
    
    template <typename T>
    Eigen::Matrix<T, Dynamic, Dynamic> toEigen(const std::vector<std::vector<T>>& stdVec) {
        
        if (stdVec.empty())
            throw NNetwork::Exception("Cannot convert empty std::vector<std::vector> to Eigen::Matrix.");
        
        size_t numRows(stdVec.size()), numCols(stdVec[0].size());
        
        // Possible return matrix
        Eigen::Matrix<T, Dynamic, Dynamic> eigenMat(numRows, numCols);
        
        // for each row
        for (int i = 0; i < numRows; i++) {
            if (stdVec[i].size() != numCols)
                throw NNetwork::Exception("Provided matrix contains variable length columns.");
            else
                eigenMat.row(i) = Eigen::toEigen(stdVec[i]);
        }
        
        return eigenMat;
    }
    
    template <typename T>
    std::vector<T> toStdVector(const Eigen::Matrix<T, Dynamic, 1>& eigenVec){
        return std::vector<T>(eigenVec.data(), eigenVec.data() + eigenVec.size());
    }
    
    
    // Append a 1 at each column head
    inline MatrixXd appendOnesAtHead(const Eigen::MatrixXd& mat) {
        MatrixXd augmentedMat(mat.rows() + 1, mat.cols());
        
        augmentedMat << Eigen::MatrixXd::Ones(1, mat.cols()), mat;
        
        return augmentedMat;
    }
    
    // Romove a particular row
    inline Eigen::MatrixXd removeRow(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                     const unsigned int rowToRemove) {
        Eigen::MatrixXd reducedMatrix = matrix;
        
        size_t numRows = matrix.rows();
        size_t numCols = matrix.cols();
        
        if( rowToRemove < numRows - 1)
            reducedMatrix.block(rowToRemove, 0 ,
                                numRows - 1 - rowToRemove,
                                numCols) = reducedMatrix.block(rowToRemove + 1, 0,
                                                               numRows- 1 -rowToRemove,
                                                               numCols);
        
        reducedMatrix.conservativeResize(numRows - 1,numCols);
        return reducedMatrix;
    }
    
    // Remove a particular column
    inline Eigen::MatrixXd removeColumn(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                        const unsigned int colToRemove) {
        Eigen::MatrixXd reducedMatrix = matrix;
        
        size_t numRows(reducedMatrix.rows()), numCols(reducedMatrix.cols());
        
        if( colToRemove < numCols - 1)
            reducedMatrix.block(0, colToRemove,
                                numRows, numCols - 1 - colToRemove) = matrix.block(0,
                                                                                   colToRemove + 1,
                                                                                   numRows, numCols - 1 - colToRemove);
        
        reducedMatrix.conservativeResize(numRows,numCols - 1);
        return reducedMatrix;
    }
    
    
    inline Eigen::MatrixXd resetColumnToValue(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                              const unsigned int colToReset) {
        Eigen::MatrixXd resetedMatrix = matrix;
        resetedMatrix.col(colToReset) = Eigen::VectorXd::Zero(resetedMatrix.rows());
        return resetedMatrix;
    }
    
    inline Eigen::MatrixXd resetRowToZero(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                          const unsigned int rowToReset) {
        Eigen::MatrixXd resetedMatrix = matrix;
        resetedMatrix.row(rowToReset) = Eigen::VectorXd::Zero(resetedMatrix.cols()).transpose();
        return resetedMatrix;
    }
    
    
    // Append given vector as a row to the existing matrix (at the bottom)
    inline void appendRowAtBottom(Eigen::MatrixXd& matrix,
                                  const Eigen::Ref<const Eigen::VectorXd>& vec) {
        
        // check consistancy with the number of cols
        if (matrix.cols() != vec.rows())
            throw NNetwork::Exception("Tried appending a vector with mis-matching dimensions (number of columns).");
        
        matrix =  ( Eigen::MatrixXd(matrix.rows() + 1, matrix.cols())
                   << matrix, vec.transpose() ).finished();
        
    }
    // Append given vector as a column to an existing matrix (on the right)
    inline void appendColumnAtRight(Eigen::MatrixXd& matrix,
                                    const Eigen::Ref<const Eigen::VectorXd>& vec) {
        
        // check consistancy with the number of cols
        if (matrix.rows() != vec.rows())
            throw NNetwork::Exception("Tried appending a vector with mis-matching dimensions (number of rows).");

        matrix = ( Eigen::MatrixXd(matrix.rows(), matrix.cols() + 1)
                  << matrix, vec ).finished();
        
    }
    
    
    template <typename MatrixType>
    inline MatrixType random(const size_t numRows, const size_t numCols,
                             const typename MatrixType::Scalar lower, const typename MatrixType::Scalar upper){
        
        return (lower * MatrixType::Ones(numRows, numCols)
                + (upper - lower) * MatrixType::Random(numRows, numCols).cwiseAbs());
        
    }
    
    // Produce a transformed vector by operating the given funcion
    inline Eigen::VectorXd operateFunction(const std::function<double(double)>& function,
                                           const Eigen::VectorXd &inputs) {
        
        size_t size = inputs.size();
        
        Eigen::VectorXd output(size);
        
        for (int i = 0; i < size; i++) {
            output[i] = function(inputs[i]);
        }
        
        return output;
    }
    
    inline Eigen::VectorXd operateFunction(const std::function<double(double, double)>& function,
                                           const Eigen::Ref<const Eigen::VectorXd>& in1,
                                           const Eigen::Ref<const Eigen::VectorXd>& in2) {
        
        size_t size = in1.size();
        
        if (size != in2.size()) {
            throw NNetwork::Exception("Input vectors size mis-match error.");
        }
        
        Eigen::VectorXd output(size);
        
        for (int i = 0; i < size; i++) {
            output[i] = function(in1[i], in2[i]);
        }
        
        return output;
    }
    
    template <typename MatrixType>
    void saveToFile(const char *filename, const Eigen::Ref<const MatrixType>& mat,
                    const std::ios::ios_base::openmode& ioType = std::ios::binary) {
        std::ofstream outFile(filename, ioType);
        typename MatrixType::Index rows(mat.rows()), cols(mat.cols());
        outFile.write((char *)&rows, sizeof(typename MatrixType::Index));
        outFile.write((char *)&cols, sizeof(typename MatrixType::Index));
        outFile.write((char *)mat.data(), sizeof(typename MatrixType::Scalar)*rows*cols);
        outFile.close();
    }
    
    template <typename MatrixScalar>
    Eigen::Matrix<MatrixScalar, Eigen::Dynamic, Eigen::Dynamic> loadFromBinaryFile(const char* filename,
                                                                                   const std::ios::ios_base::openmode& openMode = std::ios::binary) {
        typedef Eigen::Matrix<MatrixScalar, Dynamic, Dynamic> MatrixType;
        // Open file
        std::ifstream inFile(filename, openMode);
        // A matrix
        MatrixType matrix;
        
        typename MatrixType::Index rows=0, cols=0;
        inFile.read((char*) (&rows),sizeof(typename MatrixType::Index));
        inFile.read((char*) (&cols),sizeof(typename MatrixType::Index));
        matrix.resize(rows, cols);
        inFile.read( (char *) matrix.data() , rows*cols*sizeof(typename MatrixType::Scalar) );
        inFile.close();
        
        return matrix;
    }
    
    inline void fetchNextExample(Eigen::VectorXd& input,
                                 Eigen::VectorXd& output,
                                 int inputLength, int outputLength,
                                 std::ifstream& sourceFile,
                                 const std::ios::ios_base::openmode& openMode = std::ios::binary){
        
        // if the supplied file is not open, open it
        if (!sourceFile.is_open() ||
            !sourceFile.good())
            throw NNetwork::Exception("Source file is either not open or not in healthy(good) state.");
        
        input.resize(inputLength);
        output.resize(outputLength);
        
        sourceFile.read((char *)input.data(), inputLength * sizeof(double));
        sourceFile.read((char *)output.data(), outputLength * sizeof(double));
    }
    
}


#endif
