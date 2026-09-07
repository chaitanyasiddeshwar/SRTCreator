#include "audio.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
}

namespace audio {

namespace {

// Resample one decoded frame into out_pcm (mono, interleaved float == one
// float per sample). Returns false on a hard swresample error.
bool append_frame(SwrContext* swr, AVFrame* frame, std::vector<float>& out) {
    int in_samples = frame ? frame->nb_samples : 0;
    const uint8_t** in_data = frame ? (const uint8_t**)frame->extended_data : nullptr;

    int max_out = (int)swr_get_out_samples(swr, in_samples);
    if (max_out <= 0) return true;

    size_t cur = out.size();
    out.resize(cur + (size_t)max_out);
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data() + cur);

    int got = swr_convert(swr, &out_ptr, max_out, in_data, in_samples);
    if (got < 0) { out.resize(cur); return false; }
    out.resize(cur + (size_t)got);
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

    // Target: mono, opts.sample_rate, interleaved float32.
    AVChannelLayout out_ch;
    av_channel_layout_default(&out_ch, 1);

    SwrContext* swr = nullptr;
    int ret = swr_alloc_set_opts2(&swr,
                                  &out_ch, AV_SAMPLE_FMT_FLT, opts.sample_rate,
                                  &ctx->ch_layout, ctx->sample_fmt, ctx->sample_rate,
                                  0, nullptr);
    if (ret < 0 || !swr || swr_init(swr) < 0) {
        err = "failed to initialize resampler";
        if (swr) swr_free(&swr);
        av_channel_layout_uninit(&out_ch);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
        return false;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc();
    bool ok = true;

    const size_t limit_samples = opts.max_seconds > 0.0
        ? (size_t)(opts.max_seconds * opts.sample_rate) : 0;

    // Total seconds for progress: the limit if set, else the container duration.
    double total_sec = 0.0;
    if (opts.max_seconds > 0.0) total_sec = opts.max_seconds;
    else if (fmt->duration != AV_NOPTS_VALUE) total_sec = (double)fmt->duration / AV_TIME_BASE;
    int last_pct = -1;

    while (ok && av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == stream_index) {
            if (avcodec_send_packet(ctx, pkt) >= 0) {
                while (avcodec_receive_frame(ctx, frame) >= 0) {
                    if (!append_frame(swr, frame, out_pcm)) { ok = false; break; }
                }
            }
        }
        av_packet_unref(pkt);

        if (opts.on_progress && total_sec > 0.0) {
            double done = (double)out_pcm.size() / opts.sample_rate;
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
            if (!append_frame(swr, frame, out_pcm)) { ok = false; break; }
        }
    }
    // Flush resampler.
    if (ok) ok = append_frame(swr, nullptr, out_pcm);

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

} // namespace audio
