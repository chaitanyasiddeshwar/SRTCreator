cmake_minimum_required(VERSION 3.21)
project(srtcreator LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# whisper.cpp (vendored) with the CUDA backend.
# We let whisper's own CMake resolve all ggml/CUDA linkage - linking the
# `whisper` target pulls in ggml, ggml-cuda, cudart, cublas, etc. transitively.
# ---------------------------------------------------------------------------
set(GGML_CUDA              ON  CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)   # static whisper/ggml into the exe
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)   # Ampere / RTX 3080 Ti (SM 8.6)
endif()

add_subdirectory(third_party/whisper.cpp)

# MSVC 14.44 hits an Internal Compiler Error (C1001) optimizing whisper.cpp's
# heavily-inlined std::thread callable at /O2 /Ob2. Reduce inlining for the
# whisper host target only; the runtime hot path is the CUDA kernels, so this
# has no meaningful performance cost.
if(MSVC)
  target_compile_options(whisper PRIVATE /Ob1)
endif()

# ---------------------------------------------------------------------------
# FFmpeg dev/shared libraries (provided locally, see README).
# Bootstrap: link the prebuilt MSVC import libs. The LGPL minimized-static
# build (scripts/build-ffmpeg.sh) replaces this later without code changes.
# ---------------------------------------------------------------------------
set(FFMPEG_DIR "${CMAKE_SOURCE_DIR}/third_party/ffmpeg")
if(NOT EXISTS "${FFMPEG_DIR}/include")
  message(FATAL_ERROR
    "FFmpeg dev libs not found at ${FFMPEG_DIR}.\n"
    "Provide include\\, lib\\, bin\\ there (see README / CLAUDE.md §4).")
endif()

# ONNX Runtime (DirectML) for vocal isolation.
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

# Core pipeline shared by the CLI and the GUI.
add_library(srtcore STATIC
  src/audio.cpp
  src/transcribe.cpp
  src/srt.cpp
  src/models.cpp
  src/log.cpp
  src/separate.cpp
  src/debug.cpp
)
target_include_directories(srtcore PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/third_party"
  "${FFMPEG_DIR}/include"
  "${ORT_DIR}/include"
  "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include"
  "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include"
)
target_link_directories(srtcore PUBLIC "${FFMPEG_DIR}/lib" "${ORT_DIR}/lib")
target_link_libraries(srtcore PUBLIC whisper avformat avcodec avutil swresample onnxruntime)

# CLI and GUI executables.
add_executable(srt src/main.cpp)
target_link_libraries(srt PRIVATE srtcore)

add_executable(srtgui WIN32 src/gui/main_gui.cpp)
target_link_libraries(srtgui PRIVATE srtcore comctl32)

# MSVC: compile our sources as UTF-8 (so labels like -> and ... are correct),
# emit debug info even in Release, and link dbghelp for crash backtraces.
if(MSVC)
  target_compile_options(srtcore PUBLIC /utf-8 /Zi)
  target_link_libraries(srtcore PUBLIC dbghelp)
  target_link_options(srt    PRIVATE /DEBUG)
  target_link_options(srtgui PRIVATE /DEBUG)
endif()

# ---------------------------------------------------------------------------
# Stage runtime DLLs (FFmpeg + CUDA) next to each exe so they run from build\.
# ---------------------------------------------------------------------------
function(stage_runtime_dlls tgt)
  file(GLOB _ff "${FFMPEG_DIR}/bin/*.dll")
  if(_ff)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_ff} $<TARGET_FILE_DIR:${tgt}>
      COMMAND_EXPAND_LISTS)
  endif()
  file(GLOB _ort "${ORT_DIR}/bin/*.dll")   # onnxruntime.dll + DirectML.dll
  if(_ort)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_ort} $<TARGET_FILE_DIR:${tgt}>
      COMMAND_EXPAND_LISTS)
  endif()
  if(DEFINED ENV{CUDA_PATH})
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cp)
    # CUDA 12 ships these in bin\; CUDA 13 moved them to bin\x64\.
    file(GLOB _cu
      "${_cp}/bin/cudart64_*.dll"     "${_cp}/bin/cublas64_*.dll"     "${_cp}/bin/cublasLt64_*.dll"
      "${_cp}/bin/x64/cudart64_*.dll" "${_cp}/bin/x64/cublas64_*.dll" "${_cp}/bin/x64/cublasLt64_*.dll")
    if(_cu)
      add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_cu} $<TARGET_FILE_DIR:${tgt}>
        COMMAND_EXPAND_LISTS)
    endif()
  endif()
endfunction()

stage_runtime_dlls(srt)
stage_runtime_dlls(srtgui)
