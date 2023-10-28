#include "config.h"
#include "flags_handler.h"
#include "detector.h"

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

using namespace zoodemo;

int main(int argc, char* argv[]) {
    // Set precision type to F16, as that is the model variant chosen
    const ov::element::Type precision_type = ov::element::f16;

    ov::Core core;
    // core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));

    Config cfg;
    FlagsHandler flags_handler(cfg, argc, argv);
    
    if(!flags_handler.valid()) {
        return -1;
    }

    // Configure the person-vehicle-bike-detection model (model filepath, minimum confidence level)
    Model person_vehicle_bike;
    person_vehicle_bike.model_fp = cfg.pb;
    person_vehicle_bike.minimum_conf = 0.35;
    // Define the function that will be run after inference on the frame
    person_vehicle_bike.func = [](cv::Mat& frame, const Result& res) {
            cv::Scalar color;
            std::string text;

            switch(res.label_id) {
            // Vehicle bounding boxes will be blue (BGR)
            case 0: 
                color = cv::Scalar(255, 0, 0); 
                text = "vehicle";
                break;
            // Pedestrian bounding boxes will be green (BGR)
            case 1: 
                color = cv::Scalar(0, 255, 0); 
                text = "pedestrian";
                break;
            // Bicycle bounding boxes will be red (BGR)
            case 2: 
                color = cv::Scalar(0, 0, 255);
                text = "bicycle";
                break;
            default: throw std::logic_error("Invalid label applied");
            }

            cv::putText(frame, text, res.topleft, cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1);
            cv::rectangle(frame, res.topleft, res.bottomright, color, 2);
        };
    
    // Configure the vehicle-license-plate (model filepath, minimum confidence level)
    Model license_plate;
    license_plate.model_fp = cfg.lp;
    license_plate.minimum_conf = 0.1;
    // Define the function that will be run after inference on the frame
    license_plate.func = [](cv::Mat& frame, const Result& res) {
            // Only act on license plates
            if(res.label_id == 2) {
                // License plate bounding boxes will be turquoise (BGR)
                cv::Scalar color = cv::Scalar(128, 128, 0);
                cv::rectangle(frame, res.topleft, res.bottomright, color, 2);
                cv::putText(frame, "License Plate", res.topleft, cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1);
            }
        };

    Detector detector(std::make_shared<ov::Core>(core), cfg.input, { person_vehicle_bike, license_plate });
    detector.start();

    cv::destroyAllWindows();

    return 0;
}
