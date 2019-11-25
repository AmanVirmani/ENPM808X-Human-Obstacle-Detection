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
 * @file      DataLoader.cpp
 * @author    Aman Virmani (AmanVirmani) Driver
 * @author    Naman Gupta (namangupta98) Navigator
 * @author    Saumil Shah (SaumilShah66) Design Keeper
 * @copyright MIT License
 * @brief     DataLoader method implementation 
 */

#include <DataLoader.hpp>

DataLoader::DataLoader() {
  /// initialise ROI(Region of Interest) coordinates for training data
  pos_roi.x = 16;
  pos_roi.y = 16;
  pos_roi.width = 64;
  pos_roi.height = 128;
  neg_roi.x = 0;
  neg_roi.y = 0;
  neg_roi.width = 64;
  neg_roi.height = 128;
}

std::vector<cv::String> DataLoader::give_image_names(std::string dir) {
  std::vector < cv::String > names;
  cv::glob(dir, names);
  return names;
}

bool DataLoader::setupHOG() {
  hog.cellSize = cellsize;
  hog.winSize = windowSize;
  return true;
}

void DataLoader::get_training_data() {
  std::cout << "Reading directories to get names of images" << std::endl;
  auto pos_names = give_image_names(pos_dir);
  std::cout << "Number of images " << pos_names.size() << "" << std::endl;
  auto neg_names = give_image_names(neg_dir);
  std::cout << "Got the list of image names" << std::endl;
  std::vector<int> pos_labels(pos_names.size(), 0);
  std::vector<int> neg_labels(
      neg_names.size() * number_of_images_from_one_negative_image, 1);
  std::cout << "Reading positive images" << std::endl;
  for (auto name : pos_names) {
    descriptors.push_back(loadPosImage(name));
  }
  std::cout << "Positive images read successfully" << std::endl;
  std::cout << "Reading negative images" << std::endl;
  for (auto name : neg_names) {
    image = cv::imread(name);
    for (int start = 0; start < 10; start++) {
      descriptors.push_back(loadNegImage(image));
    }
  }
  std::cout << "Negative images read successfully" << std::endl;
  labels.insert(labels.end(), pos_labels.begin(), pos_labels.end());
  labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());
  std::cout << "Shuffling the training data" << std::endl;
  shuffleData();
  std::cout << "Training data is ready" << std::endl;
  std::cout << "Positive images count : " << pos_names.size() << ""
      << std::endl;
  std::cout << "Negative images count : "
      << neg_names.size() * number_of_images_from_one_negative_image << ""
      << std::endl;
}

void DataLoader::shuffleData() {
  std::vector<int> seeds;
  if (descriptors.size() != 0) {
    for (unsigned int cont = 0; cont < descriptors.size(); cont++) {
      seeds.push_back(cont);
    }
    /// randomizing the HOG descriptor orders to remove any bias while training model
    cv::randShuffle(seeds);
    std::vector < cv::Mat > descriptors_new;
    std::vector<int> new_labels;
    for (unsigned int cont = 0; cont < descriptors.size(); cont++) {
      descriptors_new.push_back(descriptors[seeds[cont]]);
      new_labels.push_back(labels[seeds[cont]]);
    }
    labels = new_labels;
    descriptors = descriptors_new;
  } else {
    std::cout << "No data available";
  }
}

cv::Mat DataLoader::loadNegImage(cv::Mat image_) {
  /// randomly choosing ROI on neg images
  neg_roi.x = rand() % (image_.size().width - windowSize.width - 1) + 1;
  neg_roi.y = rand() % (image_.size().height - windowSize.height - 1) + 1;
  crop = image_(neg_roi);
  /// converting BGR images to grayscale images to calculate HOG features
  cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
  hog.compute(gray, hog_descriptor, stride);
  return cv::Mat(hog_descriptor).clone();
}

cv::Mat DataLoader::loadPosImage(std::string imageName) {
  image = cv::imread(imageName);
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  crop = gray(pos_roi);
  /// cv::imshow("Have a look", crop);
  cv::waitKey(10);
  hog.compute(crop, hog_descriptor, stride);
  return cv::Mat(hog_descriptor).clone();
}

DataLoader::~DataLoader() {
}

