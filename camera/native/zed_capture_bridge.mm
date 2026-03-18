#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <zed_video_capture.h>

namespace {

constexpr std::size_t kStringSize = 128;

struct ZedCalibrationC {
    int stereo_width;
    int stereo_height;
    int left_width;
    int left_height;
    int channels;
    float fx;
    float fy;
    float cx;
    float cy;
    float k1;
    float k2;
    float p1;
    float p2;
    float k3;
    char serial[kStringSize];
    char name[kStringSize];
    char calibration_section[kStringSize];
};

struct ZedCameraHandle {
    zed::VideoCapture capture;
    zed::StereoDimensions stereo_dimensions;
    ZedCalibrationC calibration{};
    std::mutex mutex;
    std::vector<std::uint8_t> latest_left_bgr;
    double latest_timestamp_s = 0.0;
    bool is_open = false;
    bool is_running = false;
    bool has_frame = false;
};

double now_seconds() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

void pump_main_run_loop(double seconds) {
    if (seconds <= 0.0) {
        seconds = 0.001;
    }
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, false);
}

void copy_string(const std::string& input, char* output, std::size_t output_size) {
    if (output == nullptr || output_size == 0) {
        return;
    }
    const std::size_t n = std::min(output_size - 1, input.size());
    std::memcpy(output, input.data(), n);
    output[n] = '\0';
}

void clear_error(char* error_message, std::size_t error_size) {
    if (error_message != nullptr && error_size > 0) {
        error_message[0] = '\0';
    }
}

int write_error(const std::string& message, char* error_message, std::size_t error_size) {
    copy_string(message, error_message, error_size);
    return 0;
}

void populate_calibration(ZedCameraHandle* handle, zed::CalibrationData& calibration_data) {
    const std::string section =
        "LEFT_CAM_" + calibration_data.calibrationString(handle->stereo_dimensions);

    handle->calibration.stereo_width = static_cast<int>(handle->stereo_dimensions.width);
    handle->calibration.stereo_height = static_cast<int>(handle->stereo_dimensions.height);
    handle->calibration.left_width = static_cast<int>(handle->stereo_dimensions.width / 2);
    handle->calibration.left_height = static_cast<int>(handle->stereo_dimensions.height);
    handle->calibration.channels = 3;
    handle->calibration.fx = calibration_data.get<float>(section, "fx");
    handle->calibration.fy = calibration_data.get<float>(section, "fy");
    handle->calibration.cx = calibration_data.get<float>(section, "cx");
    handle->calibration.cy = calibration_data.get<float>(section, "cy");
    handle->calibration.k1 = calibration_data.get<float>(section, "k1");
    handle->calibration.k2 = calibration_data.get<float>(section, "k2");
    handle->calibration.p1 = calibration_data.get<float>(section, "p1");
    handle->calibration.p2 = calibration_data.get<float>(section, "p2");
    handle->calibration.k3 = calibration_data.get<float>(section, "k3");
    copy_string(handle->capture.getDeviceSerialNumber(), handle->calibration.serial, kStringSize);
    copy_string(handle->capture.getDeviceName(), handle->calibration.name, kStringSize);
    copy_string(section, handle->calibration.calibration_section, kStringSize);
}

}  // namespace

extern "C" {

ZedCameraHandle* zed_camera_create() {
    return new ZedCameraHandle();
}

void zed_camera_destroy(ZedCameraHandle* handle) {
    if (handle == nullptr) {
        return;
    }

    try {
        if (handle->is_running) {
            handle->capture.stop();
            handle->is_running = false;
        }
        if (handle->is_open) {
            handle->capture.close();
            handle->is_open = false;
        }
    } catch (...) {
    }

    delete handle;
}

int zed_camera_open_hd1080(ZedCameraHandle* handle, char* error_message, std::size_t error_size) {
    clear_error(error_message, error_size);
    if (handle == nullptr) {
        return write_error("Invalid ZED camera handle", error_message, error_size);
    }

    try {
        if (handle->is_open) {
            return 1;
        }

        handle->stereo_dimensions = handle->capture.open<zed::HD1080, zed::FPS_30>(zed::BGR);
        zed::CalibrationData calibration_data = handle->capture.getCalibrationData();
        populate_calibration(handle, calibration_data);
        handle->latest_left_bgr.assign(
            static_cast<std::size_t>(handle->calibration.left_width)
                * static_cast<std::size_t>(handle->calibration.left_height)
                * static_cast<std::size_t>(handle->calibration.channels),
            0
        );
        handle->latest_timestamp_s = 0.0;
        handle->has_frame = false;
        handle->is_open = true;
        return 1;
    } catch (const std::exception& exc) {
        return write_error(exc.what(), error_message, error_size);
    } catch (...) {
        return write_error("Unknown error while opening ZED camera", error_message, error_size);
    }
}

int zed_camera_start(ZedCameraHandle* handle, char* error_message, std::size_t error_size) {
    clear_error(error_message, error_size);
    if (handle == nullptr) {
        return write_error("Invalid ZED camera handle", error_message, error_size);
    }
    if (!handle->is_open) {
        return write_error("ZED camera must be opened before start", error_message, error_size);
    }
    if (handle->is_running) {
        return 1;
    }

    try {
        handle->capture.start([handle](std::uint8_t* data, std::size_t frame_height, std::size_t frame_width, std::size_t channels) {
            if (data == nullptr || channels != 3 || frame_width < 2) {
                return;
            }

            const std::size_t left_width = frame_width / 2;
            const std::size_t row_bytes = left_width * channels;
            std::lock_guard<std::mutex> guard(handle->mutex);
            if (handle->latest_left_bgr.size() != row_bytes * frame_height) {
                handle->latest_left_bgr.resize(row_bytes * frame_height);
            }
            for (std::size_t row = 0; row < frame_height; ++row) {
                const std::uint8_t* src = data + row * frame_width * channels;
                std::uint8_t* dst = handle->latest_left_bgr.data() + row * row_bytes;
                std::memcpy(dst, src, row_bytes);
            }
            handle->latest_timestamp_s = now_seconds();
            handle->has_frame = true;
        });
        handle->is_running = true;
        return 1;
    } catch (const std::exception& exc) {
        return write_error(exc.what(), error_message, error_size);
    } catch (...) {
        return write_error("Unknown error while starting ZED camera", error_message, error_size);
    }
}

int zed_camera_stop(ZedCameraHandle* handle, char* error_message, std::size_t error_size) {
    clear_error(error_message, error_size);
    if (handle == nullptr) {
        return write_error("Invalid ZED camera handle", error_message, error_size);
    }

    try {
        if (handle->is_running) {
            handle->capture.stop();
            handle->is_running = false;
        }
        if (handle->is_open) {
            handle->capture.close();
            handle->is_open = false;
        }
        return 1;
    } catch (const std::exception& exc) {
        return write_error(exc.what(), error_message, error_size);
    } catch (...) {
        return write_error("Unknown error while stopping ZED camera", error_message, error_size);
    }
}

int zed_camera_get_calibration(
    ZedCameraHandle* handle,
    ZedCalibrationC* calibration,
    char* error_message,
    std::size_t error_size
) {
    clear_error(error_message, error_size);
    if (handle == nullptr || calibration == nullptr) {
        return write_error("Invalid ZED calibration request", error_message, error_size);
    }
    if (!handle->is_open) {
        return write_error("ZED camera is not open", error_message, error_size);
    }
    *calibration = handle->calibration;
    return 1;
}

int zed_camera_copy_latest_left_frame(
    ZedCameraHandle* handle,
    std::uint8_t* output,
    std::size_t output_size,
    double* timestamp_s,
    char* error_message,
    std::size_t error_size
) {
    clear_error(error_message, error_size);
    if (handle == nullptr || output == nullptr || timestamp_s == nullptr) {
        return write_error("Invalid ZED frame request", error_message, error_size);
    }

    pump_main_run_loop(0.002);

    std::lock_guard<std::mutex> guard(handle->mutex);
    if (!handle->has_frame) {
        return 0;
    }
    if (output_size < handle->latest_left_bgr.size()) {
        return write_error("Output buffer is too small for the latest ZED frame", error_message, error_size);
    }

    std::memcpy(output, handle->latest_left_bgr.data(), handle->latest_left_bgr.size());
    *timestamp_s = handle->latest_timestamp_s;
    return 1;
}

int zed_camera_wait_for_frame(
    ZedCameraHandle* handle,
    int timeout_ms,
    char* error_message,
    std::size_t error_size
) {
    clear_error(error_message, error_size);
    if (handle == nullptr) {
        return write_error("Invalid ZED camera handle", error_message, error_size);
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> guard(handle->mutex);
            if (handle->has_frame) {
                return 1;
            }
        }
        pump_main_run_loop(0.005);
    }
    return 0;
}

}  // extern "C"
