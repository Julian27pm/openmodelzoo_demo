#include "detector.h"
#include "pool.h"

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace zoodemo;

Detector::Detector(std::shared_ptr<ov::Core> core, const std::string& video_fp, std::vector<Model> models)
    : _core(std::move(core)), _pool(_res), _models(std::move(models))
{
    const std::string device_type = "CPU";
    const ov::element::Type precision_type = ov::element::f16;
    const ov::preprocess::ColorFormat color_type = ov::preprocess::ColorFormat::BGR;

    // Check camera, and capture a frame to obtain dimensions
    _cap = cv::VideoCapture(video_fp);
    if(!_cap.isOpened()){
        throw std::runtime_error("Could not open video stream or file\n");
    }

    cv::Mat frame;
    _cap >> frame;

    for(auto itr : _models) {
        add_model(itr.model_fp, device_type, frame, precision_type, color_type);
    }
}

Detector::~Detector() {
    _cap.release();
}

void Detector::add_model(const std::string& model_fp, const std::string& dev, const cv::Mat& frame, ov::element::Type model_prec, ov::preprocess::ColorFormat color_format) {
    // Read model files and define preprocessing steps
    std::shared_ptr<ov::Model> model = _core->read_model(model_fp);    

    ov::preprocess::PrePostProcessor ppp(model);
    ov::preprocess::InputInfo& input = ppp.input();

    // Set input tensor format - OpenCV is int8, and in NHWC format & RGB
    input.tensor()
        .set_element_type(ov::element::u8)
        .set_shape({1, frame.rows, frame.cols, frame.channels()})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::RGB);

    // Define preprocessing, consisting of precision change, RGB->BGR and resizing
    input.preprocess()
        .convert_element_type(model_prec)
        .convert_color(color_format)
        .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    
    model = ppp.build();
    
    // Compile model, and create a worker for that model
    ov::CompiledModel compiled_model = _core->compile_model(model, dev, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    _pool.add_worker(compiled_model);
}

void Detector::start() {
    // Define lambda function to process the output tensor of each inference result
    auto draw_bb = [](const ov::Tensor& output_tensor, cv::Mat& frame, float min_conf, 
        std::function<void(cv::Mat&, const Result&)> rect_func) {

        const int detected_count = output_tensor.get_shape()[2];
        const int shape_size = output_tensor.get_shape()[3];

        // Each entry is in SSD format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        const float* tensor_data = output_tensor.data<float>();

        for(int i = 0; i < detected_count; ++i) {
            const int base = i * shape_size;
            float image_id = tensor_data[base];

            // Image ID is -1 denotes the end of image data
            if(image_id == -1) {
                break;
            }
            
            // Do not process if confidence is below set minimum
            float conf = tensor_data[base + 2];
            if(conf < min_conf) {
                continue;
            }

            Result fr_res(tensor_data[base + 1]);
            fr_res.conf = conf;

            const int x1 = static_cast<int>(tensor_data[base + 3]*frame.cols);
            const int y1 = static_cast<int>(tensor_data[base + 4]*frame.rows);
            const int x2 = static_cast<int>(tensor_data[base + 5]*frame.cols);
            const int y2 = static_cast<int>(tensor_data[base + 6]*frame.rows);

            fr_res.topleft = {x1, y1};
            fr_res.bottomright = {x2, y2};

            // Run respective function defined for the model
            rect_func(frame, fr_res);
        }
    };

    cv::Mat frame;

    while(true) {
        _cap >> frame;

        // Pass frame to pool, which will then distribute it amongst workers
        _pool.make_request(frame);
        // Wait for all requests to complete
        _pool.wait_for_all();

        // Run draw_bb lambda function for each result, and model func
        for(int i = 0; i < _res.size(); ++i) {
            draw_bb(_res[i], frame, _models[i].minimum_conf, _models[i].func);
        }

        // Display the resulting frame
        cv::imshow( "Frame", frame );
        
        // Press ESC to exit
        if((char)cv::waitKey(1) == 27) {
            break;
        }
    }
}
