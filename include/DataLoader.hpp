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
 * @file      DataLoader.hpp
 * @author    Saumil Shah (SaumilShah66) Driver
 * @author    Naman Gupta (namangupta98) Navigator
 * @author    Aman Virmani (AmanVirmani) Design Keeper
 * @copyright MIT License
 * @brief     DataLoader Class declaration
 * @detail    Declared functions Class to read data from files
 */
#ifndef INCLUDE_DATALOADER_HPP_
#define INCLUDE_DATALOADER_HPP_
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief class DataLoader
 * The class DataLoader reads data from files for training purposes and creates
 * HOG feature for every image.
 */
class DataLoader {
 public:
  /**
   * @brief Container to store no. of images from a negative image of type int 
   * 		and value 10
   */
  int number_of_images_from_one_negative_image = 10;

  /**
   * @brief constructor DataLoader
   * @param none
   * @return none
   */
  DataLoader();

  /**
   * @brief Container to store HOG features of type std::vector<cv::Mat>
   */
  std::vector<cv::Mat> descriptors;

  /**
   * @brief Container to store binary values for human detected or not as '0' 
   * 		or '1' respectively of type std::vector<int>
   */
  std::vector<int> labels;

  /**
   * @brief Function to load positive images from image files
   * @param imageName of type string
   * @return images of type cv::Mat
   * The following function loads positive images i.e. images with humans in it
   * from image files.
   */
  cv::Mat loadPosImage(std::string imageName);

  /**
   * @brief Function to load negative images from image matrix
   * @param imageNeg of type cv::Mat
   * @return images of type cv::Mat
   * The following function loads negative images i.e. images without humans in
   * it from image matrix.
   */
  cv::Mat loadNegImage(cv::Mat imageNeg);

  /**
   * @brief Function to set HOG parameters
   * @param none
   * @return none
   * This function sets the size of parameters used in HOG
   */
  virtual void setupHOG();

  /**
   * @brief Function to get data for training
   * @param none
   * @return none
   * The function takes data from the directory to train SVM
   */
  virtual void get_training_data();

  /**
   * @brief destructor DataLoader
   * @param none
   * @return none
   */
  virtual ~DataLoader();

  /**
   * @brief Gives images names from directory
   * @param dir of type string
   * @return container of type std::vector<cv::String>
   */
  virtual std::vector<cv::String> give_image_names(std::string dir);

  /**
   * @brief Container for positive directory with initial value as 0 for empty
   * 		directory of type string
   */
  std::string pos_dir = "0";

  /**
   * @brief Container for positive directory with initial value as 0 for empty
   * 		directory of type string
   */
  std::string neg_dir = "0";

 private:
  /**
   * @brief Instance of HOG in which HOG feature is created of type 
   * 		cv::HOGDescriptor
   */
  cv::HOGDescriptor hog;

  /**
   * @brief Container to store colored image of type cv::Mat
   */
  cv::Mat image;

  /**
   * @brief Container to store grayscale image of type cv::Mat
   */
  cv::Mat gray;

  /**
   * @brief Container to store cropped image of type cv::Mat
   */
  cv::Mat crop;

  /**
   * @brief Instance in which captured video is stored of type cv::VideoCapture
   */
  cv::VideoCapture cap;

  /**
   * @brief Container to store HOG features of an image of type 
   * 		std::vector<float>
   */
  std::vector<float> hog_descriptor;

  /**
   * @brief Container to store cell size of kernel feature of HOG of type 
   * 		cv::Size and value cv::Size(4,4)
   */
  cv::Size cellsize = cv::Size(4, 4);

  /**
   * @brief Container to store image size of type cv::Size and value 
   * 		cv::Size(64,128)
   */
  cv::Size rsize = cv::Size(64, 128);

  /**
   * @brief Container to stride of type cv::Size and value cv::Size(8,8)
   */
  cv::Size stride = cv::Size(8, 8);

  /**
   * @brief Container to store window size of type cv::Size	and value 
   * 		cv::Size(64,128)
   */
  cv::Size windowSize = cv::Size(64, 128);

  /**
   * @brief Function to shuffle data
   * @param none
   * @return none
   * Function shuffles data which is required for training to get better result
   */
  void shuffleData();

  /**
   * @brief Container stores negative region of interest data of type cv::Rect
   */
  cv::Rect neg_roi;

  /**
   * @brief Container stores positive region of interest data of type cv::Rect
   */
  cv::Rect pos_roi;

};

#endif  // INCLUDE_DATALOADER_HPP_
