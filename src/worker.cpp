#include "worker.h"
#include "detector.h"
#include "pool.h"

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <string>
#include <vector>

using namespace zoodemo;

Worker::Worker(ov::CompiledModel& compiled_model, std::vector<ov::Tensor>& res, int id) 
    : _compiled_model(compiled_model), _res(res), _busy(false), _id(id)
{
    _input_port = _compiled_model.input();
    _infer_request = new ov::InferRequest(_compiled_model.create_infer_request());
    
    // Set callback function for async request
    _infer_request->set_callback([this](std::exception_ptr ex) {
        if(ex) {
            std::rethrow_exception(ex);
        }
        
        _res[_id] = std::move(_infer_request->get_output_tensor());
        _busy = false;
    });
}
Worker::~Worker() {
    // Clean up raw pointer
    delete _infer_request;
    _infer_request = nullptr;
}

void Worker::make_request(const cv::Mat& frame) {
    // Create input_tensor and prepare request
    ov::Tensor input_tensor(_input_port.get_element_type(), _input_port.get_shape(), frame.data);
    _infer_request->set_input_tensor(input_tensor);

    // Set current worker status to busy
    _busy = true;
    
    // Start async request
    _infer_request->start_async();
}

void Worker::wait() {
    // If busy, wait
    if(_busy) {
        _infer_request->wait();
    }
}
