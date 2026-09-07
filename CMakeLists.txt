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

add_executable(srt
  src/main.cpp
  src/audio.cpp
  src/transcribe.cpp
  src/srt.cpp
  src/models.cpp
)

target_include_directories(srt PRIVATE
  "${FFMPEG_DIR}/include"
  "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include"
  "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include"
)
target_link_directories(srt PRIVATE "${FFMPEG_DIR}/lib")
target_link_libraries(srt PRIVATE whisper avformat avcodec avutil swresample)

# ---------------------------------------------------------------------------
# Stage runtime DLLs next to srt.exe so it runs from the build dir.
# ---------------------------------------------------------------------------
file(GLOB FFMPEG_DLLS "${FFMPEG_DIR}/bin/*.dll")
if(FFMPEG_DLLS)
  add_custom_command(TARGET srt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} $<TARGET_FILE_DIR:srt>
    COMMAND_EXPAND_LISTS)
endif()

# CUDA runtime DLLs (cudart / cublas / cublasLt) - shipped alongside the exe.
if(DEFINED ENV{CUDA_PATH})
  file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_path)
  # CUDA 12 ships these in bin\; CUDA 13 moved them to bin\x64\.
  file(GLOB CUDA_DLLS
    "${_cuda_path}/bin/cudart64_*.dll"
    "${_cuda_path}/bin/cublas64_*.dll"
    "${_cuda_path}/bin/cublasLt64_*.dll"
    "${_cuda_path}/bin/x64/cudart64_*.dll"
    "${_cuda_path}/bin/x64/cublas64_*.dll"
    "${_cuda_path}/bin/x64/cublasLt64_*.dll")
  if(CUDA_DLLS)
    add_custom_command(TARGET srt POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDA_DLLS} $<TARGET_FILE_DIR:srt>
      COMMAND_EXPAND_LISTS)
  endif()
endif()
