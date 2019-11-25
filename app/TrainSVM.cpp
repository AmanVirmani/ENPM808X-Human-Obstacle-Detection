/******************************************************************************
 * MIT License

 * Copyright (c) 2019 Saumil Shah, Aman Virmani, Naman Gupta

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. 
 ******************************************************************************/

/**
 * @file      TrainSVM.cpp
 * @author    Aman Virmani (AmanVirmani) Driver
 * @author    Naman Gupta (namangupta98) Navigator
 * @author    Saumil Shah (SaumilShah66) Design Keeper
 * @copyright MIT License
 * @brief     TrainSVM Class declaration
 * @detail    Implementation of class methods to train SVM using data from class
 * DataLoader
 */

#include <TrainSVM.hpp>

TrainSVM::TrainSVM() {
  /// initialising SVM model and setting up default values for the parameters
  svm = cv::ml::SVM::create();
  svm->setCoef0(0.0);
  svm->setDegree(3);
  svm->setTermCriteria(
      cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000,
                       1e-3));
  svm->setGamma(0);
  svm->setKernel(cv::ml::SVM::LINEAR);
  svm->setNu(0.5);
  svm->setP(0.1);  /// for EPSILON_SVR, epsilon in loss function?
  svm->setC(0.01);  /// From paper, soft classifier
  svm->setType(cv::ml::SVM::C_SVC);
}

cv::Mat TrainSVM::prepareData() {
  const int rows = static_cast<int>(descriptors.size());
  const int cols = static_cast<int>(std::max(descriptors[0].cols,
                                             descriptors[0].rows));
  /// temp is used to store data values in the format as required to train the SVM model
  cv::Mat temp(1, cols, CV_32FC1), trainData(rows, cols, CV_32FC1);
  for (unsigned int index = 0; index < descriptors.size(); index++) {
    if (descriptors[index].cols == 1) {
      transpose(descriptors[index], temp);
      temp.copyTo(trainData.row(static_cast<int>(index)));
    } else if (descriptors[index].rows == 1) {
      descriptors[index].copyTo(
          trainData.row(static_cast<int>(index)));
    }
  }
  return trainData;
}

void TrainSVM::startTraining() {
  get_training_data();
  if (descriptors.size() > 0 && labels.size() > 0) {
    std::cout << "Starting training" << std::endl;
    svm->train(prepareData(), cv::ml::ROW_SAMPLE, labels);
    std::cout << "SVM trained" << std::endl;
  } else {
    std::cout << "No data available to train SVM" << std::endl
        << "Please check the given paths for positive and negative images"
        << std::endl;
  }
}

void TrainSVM::saveSVM(std::string filename) {
  if (descriptors.size() > 0 && labels.size() > 0) {
    std::cout << "Saving trained SVM classifier" << std::endl;
    svm->save(filename);
    std::cout << "SVM classifier saved successfully" << std::endl;
  } else {
    std::cout << "No SVM trained..Nothing to save" << std::endl;
  }
}

bool TrainSVM::directoryExist(const std::string& pathname) {
  struct stat info;
  return (stat(pathname.c_str(), &info) == 0);
}

void TrainSVM::setPosDirectory(std::string dir) {
  if (directoryExist(dir)) {
    pos_dir = dir;
  } else {
    std::cout << "Positive data directory does not exists" << std::endl;
  }
}

void TrainSVM::setNegDirectory(std::string dir) {
  if (directoryExist(dir)) {
    neg_dir = dir;
  } else {
    std::cout << "Negative data directory does not exists" << std::endl;
  }
}

TrainSVM::~TrainSVM() {
}
