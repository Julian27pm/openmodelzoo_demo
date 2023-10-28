#include "pool.h"

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <memory>
#include <mutex>
#include <vector>

using namespace zoodemo;

Pool::Pool(std::vector<ov::Tensor>& res) 
    : _res(res)
{
}

void Pool::add_worker(ov::CompiledModel& compiled_model) {
    // Create worker
    std::unique_ptr<Worker> worker = std::make_unique<Worker>(compiled_model, _res, _workers.size());
    
    // Resize res vector if another worker is added
    _res.resize(_res.size()+1);
    _workers.push_back(std::move(worker));
}

void Pool::make_request(const cv::Mat& frame) {
    // Distribute frame amongst workers
    for(int i = 0; i < _workers.size(); ++i) {
        _workers[i]->make_request(frame);
    }
}

void Pool::wait_for_all() {
    // Wait for all busy workers to finish
    for(int i = 0; i < _workers.size(); ++i) {
        _workers[i]->wait();
    }
}
