#ifndef DETECTOR_H
#define DETECTOR_H

#include "pool.h"

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <memory>
#include <string>
#include <vector>

namespace zoodemo {

struct Result {
    Result(float label) : label_id(static_cast<int>(label)) {}

    int label_id;
    float conf;
    cv::Point topleft;
    cv::Point bottomright;
};

struct Model {
    std::string model_fp;
    std::function<void(cv::Mat&, const Result&)> func;
    float minimum_conf;
};

class Detector {
public:
    Detector(std::shared_ptr<ov::Core> core, const std::string& video_fp, std::vector<Model> models);
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    void add_model(const std::string& model_fp, const std::string& dev, const cv::Mat& frame, ov::element::Type model_prec, ov::preprocess::ColorFormat color_format);
    void start();

private:
    cv::VideoCapture _cap;
    std::shared_ptr<ov::Core> _core;

    Pool _pool;
    std::vector<ov::Tensor> _res;
    std::vector<std::function<void(cv::Mat&, const Result&)>> _funcs;
    std::vector<Model> _models;

    void detect(const cv::Mat& frame);
};

}   //namespace zoodemo

#endif
