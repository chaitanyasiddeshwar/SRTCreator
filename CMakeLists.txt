cmake_minimum_required(VERSION 3.21)
project(srtcreator LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# whisper.cpp / ggml consumed as PREBUILT backend-DL DLLs (mirrors FFmpeg/ONNX).
# The coherent DLL set (whisper.dll, ggml*.dll, ggml-cpu-*.dll, ggml-cuda.dll,
# ggml-vulkan.dll + import libs + headers) is produced by the separate deps build
# scripts/build-whisper-dlls.ps1 (CUDA Toolkit + Vulkan SDK; build machine / CI
# only) and staged here. This app build is MSVC-only: NO CUDA/nvcc/Vulkan, no
# add_subdirectory, no kernel compile. See modular_proposal.md 11.
# ---------------------------------------------------------------------------
set(WHISPER_DIR "${CMAKE_SOURCE_DIR}/third_party/whisper")
if(NOT EXISTS "${WHISPER_DIR}/include/whisper.h")
  message(FATAL_ERROR
    "Prebuilt whisper not found at ${WHISPER_DIR}.\n"
    "Run scripts/build-whisper-dlls.ps1 to produce include\\, lib\\, bin\\ "
    "(needs CUDA Toolkit + Vulkan SDK; see modular_proposal.md §11).")
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
  src/segment_mode.cpp
)
target_include_directories(srtcore PUBLIC
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/third_party"
  "${FFMPEG_DIR}/include"
  "${ORT_DIR}/include"
  "${WHISPER_DIR}/include"
)
target_link_directories(srtcore PUBLIC "${FFMPEG_DIR}/lib" "${ORT_DIR}/lib" "${WHISPER_DIR}/lib")
# whisper.lib -> whisper.dll (transcription). transcribe.cpp's backend/device
# queries split across two ggml DLLs: ggml.dll (ggml_backend_load_all, dev_count/
# get) and ggml-base.dll (dev_name/description/type, backend_reg, reg_name).
target_link_libraries(srtcore PUBLIC whisper ggml ggml-base avformat avcodec avutil swresample onnxruntime)

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
# Stage runtime DLLs (whisper/ggml backends + FFmpeg + ONNX + CUDA runtime) next
# to each exe so it runs from build\. ggml_backend_load_all() picks the best of
# the ggml-*.dll backends present at runtime (CUDA -> Vulkan -> CPU).
# ---------------------------------------------------------------------------
function(stage_runtime_dlls tgt)
  file(GLOB _wh "${WHISPER_DIR}/bin/*.dll")   # whisper.dll, ggml*.dll, ggml-cpu-*/cuda/vulkan
  if(_wh)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_wh} $<TARGET_FILE_DIR:${tgt}>
      COMMAND_EXPAND_LISTS)
  endif()
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
