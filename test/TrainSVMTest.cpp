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
 * @file      TrainSVMTest.cpp
 * @author    Saumil Shah (SaumilShah66) Driver
 * @author    Naman Gupta (namangupta98) Navigator
 * @author    Aman Virmani (AmanVirmani) Design Keeper
 * @copyright MIT License
 * @brief     Test cases for TrainSVM class methods
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdio.h>
#include <TrainSVM.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

/** 
 * @brief Mock Class for Dataloader
 */

class MockDataLoader : public DataLoader {
 public:
  MOCK_METHOD0(setupHOG, bool ());

  MOCK_METHOD1(get_training_data, void ());
};

/** 
 * @brief Test for Mock Class
 * It checks if DataLoader class is successfully mocked
 */

TEST(SvmTest, MockTest) {
    TrainSVM svm_trainer;
    EXPECT_CALL(svm_trainer, setupHOG()).Times(1).WillOnce(Return(true));
    EXPECT_EQ(svm_trainer.prepareData(),true);
}

/** 
 * @brief Test for SVM training
 * It checks SVMTrainer class must not train with empty given directory
 */
TEST(SVMCheck, ShouldNotTrain) {
  TrainSVM svm_trainer;
  svm_trainer.setPosDirectory("../test_data/npp");
  svm_trainer.setNegDirectory("../test_data/nnp");
  svm_trainer.startTraining();
  std::string path = "../test_data/trained"+std::to_string(std::rand())+".xml";
  svm_trainer.saveSVM(path);
  ASSERT_EQ(svm_trainer.directoryExist(path), 0);
}

/** 
 * @brief Test for SVM training
 * It checks whether SVMTrainer class has trained a classifier and saved it or not
 */
TEST(SVMCheck, ShouldTrain) {
  TrainSVM svm_trainer;
  svm_trainer.setPosDirectory("../test_data/np");
  svm_trainer.setNegDirectory("../test_data/nn");
  svm_trainer.startTraining();
  std::string path = "../test_data/trained"+std::to_string(std::rand())+".xml";
  svm_trainer.saveSVM(path);
  ASSERT_EQ(svm_trainer.directoryExist(path), 1);
  if ( remove(path.c_str() ) == 0 ) {
    std::cout << path << " deleted\n";
  }
}

