
#pragma once
#include <fstream>
#include <string>
#include <thread>
#include <sstream>

class Logger {
public:
    static void logToFile(const std::string& msg, const std::string& filename = "stab_log.txt") {
        std::ofstream logFile(filename, std::ios::app);
        if (logFile.is_open()) {
            std::ostringstream oss;
            oss << "[Thread " << std::this_thread::get_id() << "] " << msg << std::endl;
            logFile << oss.str();
        }
    }
};
