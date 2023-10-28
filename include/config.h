#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace zoodemo {

struct Config {
    std::string input = "cars.mp4";
    std::string pb = "intel/person-vehicle-bike-detection-2002/FP16/person-vehicle-bike-detection-2002.xml";
    std::string lp = "intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml";
};

}   //namespace zoodemo

#endif
