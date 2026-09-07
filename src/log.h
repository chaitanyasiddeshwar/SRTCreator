#pragma once
#include <string>

// Minimal thread-safe file logger shared by the CLI and GUI. Every line is
// flushed immediately so a crash still leaves a trace. init() also installs an
// unhandled-exception / terminate handler that logs a backtrace before dying.
namespace logging {

// Open (append) the log file and install crash handlers. `app_tag` labels lines
// ("cli"/"gui"). Safe to call once at startup.
void init(const char* app_tag);

void info (const std::string& msg);
void error(const std::string& msg);
void logf (const char* level, const char* fmt, ...);

// Absolute path of the active log file (for surfacing to the user).
std::string log_path();

} // namespace logging
