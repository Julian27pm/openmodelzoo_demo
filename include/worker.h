#ifndef WORKER_H
#define WORKER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <vector>

namespace zoodemo {

class Worker {
public:
    Worker(ov::CompiledModel& compiled_model, std::vector<ov::Tensor>& res, int id);
    ~Worker();
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    
    void make_request(const cv::Mat& frame);
    void wait();   

private:
    std::vector<ov::Tensor>& _res;

    bool _busy;
    ov::CompiledModel& _compiled_model;
    ov::InferRequest* _infer_request;
    ov::Output<const ov::Node> _input_port;

    int _id;
};

}   //namespace zoodemo

#endif
