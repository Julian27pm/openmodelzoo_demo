#include "flags_handler.h"
#include "config.h"

#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace zoodemo;

FlagsHandler::FlagsHandler(Config& cfg, int argc, char* argv[]) 
    : _cfg(cfg), _valid(false) 
{
    // If no extra flags are provided, return
    if(argc == 1) {
        _valid = true;
        return;
    }

    // Define available flags, and their assigning variables
    static const std::unordered_map<std::string, std::pair<std::string, std::string&>> assign_map {
        { "--in",  { "input video", _cfg.input } },
        { "--pb",  { "person-vehicle-bike-detection xml path", _cfg.pb } },
        { "--lp",  { "vehicle-license-plate-detection-barrier xml path", _cfg.lp } }
    };

    // Print out possible flags and their descriptions when --help is given
    if(argc == 2 && std::string(argv[1]) == "--help") {
        for(auto itr : assign_map) {
            std::cout << "  " << itr.first << " <" << itr.second.first << ">\n";
        }
        return;
    }

    // Assign values to Config based on input flags
    for(int i = 1; i < argc; i+=2) {
        std::string curr_arg = std::string(argv[i]);

        if(assign_map.find(curr_arg) == assign_map.end()) {
            throw std::invalid_argument(curr_arg + " is an invalid flag. Use --help");
        }

        if(i+1 == argc) {
            throw std::invalid_argument(curr_arg + " does not have an argument. Use --help");
        }

        assign_map.at(curr_arg).second = std::string(argv[i+1]);
    }

    _valid = true;
}

bool FlagsHandler::valid() {
    return _valid;
}
