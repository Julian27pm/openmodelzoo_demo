#ifndef FLAGS_HANDLER_H
#define FLAGS_HANDLER_H

#include "config.h"

#include <string>

namespace zoodemo {

class FlagsHandler {
public:
    FlagsHandler(Config& cfg, int argc, char* argv[]);

    bool valid();

private:
    Config& _cfg;
    bool _valid;
};

}   //namespace zoodemo

#endif
