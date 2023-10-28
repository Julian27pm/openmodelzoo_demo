#ifndef POOL_H
#define POOL_H

#include "worker.h"

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <memory>
#include <vector>

namespace zoodemo {

class Pool {
public:
    Pool(std::vector<ov::Tensor>& res);
    void add_worker(ov::CompiledModel& compiled_model);
    void make_request(const cv::Mat& frame);
    void wait_for_all();

private:
    std::vector<std::unique_ptr<Worker>> _workers;

    std::vector<ov::Tensor>& _res;
};

}   //namespace zoodemo

#endif
