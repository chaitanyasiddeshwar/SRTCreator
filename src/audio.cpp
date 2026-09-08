#include "audio.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
#include <libswresample/swresample.h>
}

namespace audio {

namespace {

// Resample one decoded frame into out_pcm (mono, interleaved float == one
// float per sample). Returns false on a hard swresample error.
bool append_frame(SwrContext* swr, AVFrame* frame, int out_nch, std::vector<float>& out) {
    int in_samples = frame ? frame->nb_samples : 0;
    const uint8_t** in_data = frame ? (const uint8_t**)frame->extended_data : nullptr;

    int max_out = (int)swr_get_out_samples(swr, in_samples);
    if (max_out <= 0) return true;

    size_t cur = out.size();
    out.resize(cur + (size_t)max_out * out_nch);
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data() + cur);

    int got = swr_convert(swr, &out_ptr, max_out, in_data, in_samples);
    if (got < 0) { out.resize(cur); return false; }
    out.resize(cur + (size_t)got * out_nch);
    return true;
}

} // namespace

bool decode(const std::string& path, const DecodeOptions& opts,
            std::vector<float>& out_pcm, std::string& err) {
    out_pcm.clear();

    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) {
        err = "could not open input: " + path;
        return false;
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        err = "could not read stream info from: " + path;
        avformat_close_input(&fmt);
        return false;
    }

    const AVCodec* dec = nullptr;
    int stream_index = opts.stream_index;
    if (stream_index < 0) {
        stream_index = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
    } else {
        if (stream_index >= (int)fmt->nb_streams ||
            fmt->streams[stream_index]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
            err = "requested --audio-stream is not an audio stream";
            avformat_close_input(&fmt);
            return false;
        }
        dec = avcodec_find_decoder(fmt->streams[stream_index]->codecpar->codec_id);
    }
    if (stream_index < 0 || !dec) {
        err = "no decodable audio stream found in: " + path;
        avformat_close_input(&fmt);
        return false;
    }

    AVStream* st = fmt->streams[stream_index];
    AVCodecContext* ctx = avcodec_alloc_context3(dec);
    if (!ctx) { err = "failed to allocate codec context"; avformat_close_input(&fmt); return false; }
    avcodec_parameters_to_context(ctx, st->codecpar);
    if (avcodec_open2(ctx, dec, nullptr) < 0) {
        err = "could not open audio decoder";
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
        return false;
    }

    // Target layout: opts.channels (1 or 2), opts.sample_rate, interleaved float.
    int out_nch = opts.channels < 1 ? 1 : opts.channels;
    AVChannelLayout out_ch;
    av_channel_layout_default(&out_ch, out_nch);

    // Dialogue lives in the Front-Center channel of a 5.1/7.1 mix. When asked,
    // route ONLY that channel into every output channel (via a rematrix matrix)
    // instead of a downmix that folds music/effects from every channel in.
    int in_ch = ctx->ch_layout.nb_channels;
    int fc_index = opts.center_channel_only
        ? av_channel_layout_index_from_channel(&ctx->ch_layout, AV_CHAN_FRONT_CENTER)
        : -1;

    SwrContext* swr = nullptr;
    int ret = swr_alloc_set_opts2(&swr,
                                  &out_ch, AV_SAMPLE_FMT_FLT, opts.sample_rate,
                                  &ctx->ch_layout, ctx->sample_fmt, ctx->sample_rate,
                                  0, nullptr);
    if (ret < 0 || !swr) {
        err = "failed to allocate resampler";
        if (swr) swr_free(&swr);
        av_channel_layout_uninit(&out_ch);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
        return false;
    }

    if (fc_index >= 0) {
        std::vector<double> matrix((size_t)out_nch * in_ch, 0.0);
        for (int o = 0; o < out_nch; ++o) matrix[(size_t)o * in_ch + fc_index] = 1.0;
        swr_set_matrix(swr, matrix.data(), in_ch);
        logging::logf("INFO", "audio: center channel -> %d ch (fc idx %d of %d)", out_nch, fc_index, in_ch);
    } else if (opts.center_channel_only) {
        logging::logf("INFO", "audio: no center channel (%d ch) - downmixing", in_ch);
    }

    if (swr_init(swr) < 0) {
        err = "failed to initialize resampler";
        swr_free(&swr);
        av_channel_layout_uninit(&out_ch);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
        return false;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc();
    bool ok = true;

    const size_t limit_samples = opts.max_seconds > 0.0
        ? (size_t)(opts.max_seconds * opts.sample_rate) * out_nch : 0;

    // Total seconds for progress: the limit if set, else the container duration.
    double total_sec = 0.0;
    if (opts.max_seconds > 0.0) total_sec = opts.max_seconds;
    else if (fmt->duration != AV_NOPTS_VALUE) total_sec = (double)fmt->duration / AV_TIME_BASE;
    int last_pct = -1;

    while (ok && av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == stream_index) {
            if (avcodec_send_packet(ctx, pkt) >= 0) {
                while (avcodec_receive_frame(ctx, frame) >= 0) {
                    if (!append_frame(swr, frame, out_nch, out_pcm)) { ok = false; break; }
                }
            }
        }
        av_packet_unref(pkt);

        if (opts.on_progress && total_sec > 0.0) {
            double done = (double)out_pcm.size() / ((double)opts.sample_rate * out_nch);
            int pct = (int)(done / total_sec * 100.0);
            if (pct > 100) pct = 100;
            if (pct != last_pct) { last_pct = pct; opts.on_progress(pct); }
        }
        if (limit_samples && out_pcm.size() >= limit_samples) break;
    }

    // Flush decoder.
    if (ok) {
        avcodec_send_packet(ctx, nullptr);
        while (avcodec_receive_frame(ctx, frame) >= 0) {
            if (!append_frame(swr, frame, out_nch, out_pcm)) { ok = false; break; }
        }
    }
    // Flush resampler.
    if (ok) ok = append_frame(swr, nullptr, out_nch, out_pcm);

    if (!ok) err = "audio decode/resample failed";

    av_frame_free(&frame);
    av_packet_free(&pkt);
    swr_free(&swr);
    av_channel_layout_uninit(&out_ch);
    avcodec_free_context(&ctx);
    avformat_close_input(&fmt);

    if (ok && out_pcm.empty()) { err = "no audio samples decoded"; return false; }
    return ok;
}

bool resample_to_mono(const std::vector<float>& in, int in_rate, int in_channels,
                      int out_rate, std::vector<float>& out, std::string& err) {
    out.clear();
    if (in.empty()) return true;

    AVChannelLayout in_ch, out_ch;
    av_channel_layout_default(&in_ch, in_channels);
    av_channel_layout_default(&out_ch, 1);

    SwrContext* swr = nullptr;
    if (swr_alloc_set_opts2(&swr, &out_ch, AV_SAMPLE_FMT_FLT, out_rate,
                            &in_ch, AV_SAMPLE_FMT_FLT, in_rate, 0, nullptr) < 0
        || !swr || swr_init(swr) < 0) {
        err = "resampler init failed";
        if (swr) swr_free(&swr);
        av_channel_layout_uninit(&in_ch);
        av_channel_layout_uninit(&out_ch);
        return false;
    }

    // Feed the input in blocks: a single swr_convert call with hundreds of
    // millions of samples (e.g. a 2-hour vocal stem) does not work reliably.
    const size_t total_in = in.size() / in_channels;
    const size_t BLOCK = 1u << 20; // input frames per call
    bool ok = true;
    for (size_t pos = 0; pos < total_in; pos += BLOCK) {
        int n = (int)std::min(BLOCK, total_in - pos);
        int max_out = (int)av_rescale_rnd(swr_get_delay(swr, in_rate) + n,
                                          out_rate, in_rate, AV_ROUND_UP) + 16;
        size_t cur = out.size();
        out.resize(cur + (size_t)max_out);
        const uint8_t* ip = reinterpret_cast<const uint8_t*>(in.data() + pos * in_channels);
        uint8_t* op = reinterpret_cast<uint8_t*>(out.data() + cur);
        int got = swr_convert(swr, &op, max_out, &ip, n);
        if (got < 0) { ok = false; out.resize(cur); break; }
        out.resize(cur + (size_t)(got > 0 ? got : 0));
    }
    // Flush.
    while (ok) {
        int max_out = (int)av_rescale_rnd(swr_get_delay(swr, in_rate), out_rate, in_rate, AV_ROUND_UP) + 16;
        size_t cur = out.size();
        out.resize(cur + (size_t)max_out);
        uint8_t* op = reinterpret_cast<uint8_t*>(out.data() + cur);
        int g = swr_convert(swr, &op, max_out, nullptr, 0);
        out.resize(cur + (size_t)(g > 0 ? g : 0));
        if (g <= 0) break;
    }

    swr_free(&swr);
    av_channel_layout_uninit(&in_ch);
    av_channel_layout_uninit(&out_ch);
    if (!ok || out.empty()) { err = "resample produced no output"; return false; }
    return true;
}

bool write_wav(const std::string& path, const std::vector<float>& pcm,
               int sample_rate, int channels, std::string& err) {
    if (channels < 1) channels = 1;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "could not open wav for writing: " + path; return false; }

    const uint32_t n_samples   = (uint32_t)pcm.size();          // total interleaved samples
    const uint16_t bits        = 16;
    const uint16_t block_align = (uint16_t)(channels * bits / 8);
    const uint32_t byte_rate   = (uint32_t)sample_rate * block_align;
    const uint32_t data_bytes  = n_samples * (bits / 8);
    const uint32_t riff_bytes  = 36 + data_bytes;

    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };

    std::fwrite("RIFF", 1, 4, f); w32(riff_bytes); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(16); w16(1 /*PCM*/); w16((uint16_t)channels);
    w32((uint32_t)sample_rate); w32(byte_rate); w16(block_align); w16(bits);
    std::fwrite("data", 1, 4, f); w32(data_bytes);

    // Convert float [-1,1] -> int16, in modest blocks to bound memory.
    std::vector<int16_t> buf;
    const size_t BLOCK = 1u << 16;
    buf.reserve(std::min<size_t>(BLOCK, pcm.size()));
    for (size_t i = 0; i < pcm.size(); ) {
        buf.clear();
        size_t end = std::min(i + BLOCK, pcm.size());
        for (; i < end; ++i) {
            float s = pcm[i];
            if (s >  1.0f) s =  1.0f;
            if (s < -1.0f) s = -1.0f;
            buf.push_back((int16_t)std::lround(s * 32767.0f));
        }
        std::fwrite(buf.data(), sizeof(int16_t), buf.size(), f);
    }
    std::fclose(f);
    return true;
}

} // namespace audio
