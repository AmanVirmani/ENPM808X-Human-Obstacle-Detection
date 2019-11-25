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
 * @file      TrainSVM.hpp
 * @author    Saumil Shah (SaumilShah66) Driver
 * @author    Naman Gupta (namangupta98) Navigator
 * @author    Aman Virmani (AmanVirmani) Design Keeper
 * @copyright MIT License
 * @brief     TrainSVM Class declaration
 * @detail    Declared functions Class to train SVM using data from class 
 *            DataLoader
 */

#ifndef INCLUDE_TRAINSVM_HPP_
#define INCLUDE_TRAINSVM_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <DataLoader.hpp>
#include <sys/stat.h>
#include <fstream>
#include <sys/types.h>


/**
 * @brief class TrainSVM
 * The class TrainSVM trains SVM using data from class DataLoader and returns 
 * the trained data into a XML file
 */
class TrainSVM : public DataLoader {
 public:
  /**
   * @brief Instance to store parameters needed for SVM of type
   *        cv::Ptr<cv::ml::SVM>
   */
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

  /**
   * @brief Instance to take data from class DataLoader
   */
//  DataLoader loadData;

  /**
   * @brief constructor TrainSVM
   * @param none
   * @return none
   */
  TrainSVM();

  /**
   * @brief Function to save trained SVM
   * @param filename of type string
   * @return none
   * The following function saves trained SVM in a file and takes the filename 
   */
  void saveSVM(std::string filename);

  /**
   * @brief Function to start training
   * @param none
   * @return none
   * The following function initiates SVM training
   */
  void startTraining();

  /**
   * @brief Function to change data format into format used by OpenCV's SVM 
   *        train method
   * @param filename of type string
   * @return training data of type cv::Mat
   */
  cv::Mat prepareData();

  /**
   * @brief destructor TrainSVM
   * @param none
   * @return none
   */
  ~TrainSVM();

  /**
   * @brief Function to check existing directory
   * @param pathname of type string
   * @return true or false as bool value
   * The following function checks either the file exist or not and returns
   * 'true' if exist and 'false' if not
   */
  bool directoryExist(const std::string& pathname);

  /**
   * @brief Function to set directory for positive images
   * @param dir of type string
   * @return none
   * The following function saves the positive images in a directory
   */
  void setPosDirectory(std::string dir);

  /**
   * @brief Function to set directory for negative images
   * @param dir of type string
   * @return none
   * The following function saves the negative images in a directory
   */
  void setNegDirectory(std::string dir);
};

#endif  // INCLUDE_DATALOADER_HPP_
