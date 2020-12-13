#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstring>

#define RESIZE_SIZE 256
#define NET_SIZE 224

void resize_and_crop(const cv::Mat& src, cv::Mat& dst) {
    const cv::Size org_size = src.size();
    cv::Size resize_size;
    if (org_size.height > org_size.width) {
        resize_size.width = RESIZE_SIZE;
        resize_size.height = static_cast<int>((float)org_size.height / org_size.width * RESIZE_SIZE);
    } else {
        resize_size.height = RESIZE_SIZE;
        resize_size.width = static_cast<int>((float)org_size.width / org_size.height * RESIZE_SIZE);
    }
    // リサイズ
    cv::Mat resize;
    cv::resize(src, resize, resize_size);
    // クロップ
    cv::Rect rect = {
        (resize_size.width - NET_SIZE) / 2,
        (resize_size.height - NET_SIZE) / 2,
        NET_SIZE,
        NET_SIZE
    };
    cv::Mat roi = resize(rect);
    roi.copyTo(dst);
}

void mat_to_tensor(const cv::Mat& src, torch::Tensor& tensor, bool gpu) {
    const cv::Size size = src.size();
    const int channel = src.channels();
    // B, G, Rにチャンネル分割
    std::vector<cv::Mat> split_src;
    cv::split(src, split_src);
    auto op = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .requires_grad(false);
    if (gpu) {
        op = op.device(torch::kCUDA);
        tensor = torch::zeros({ channel, size.height, size.width }, op);
        for (auto i = 0; i < split_src.size(); ++i) {
            cudaMemcpyAsync(
                tensor.data_ptr() + (size.width * size.height * sizeof(unsigned char) * i),
                split_src[i].data,
                sizeof(unsigned char) * size.width * size.height,
                cudaMemcpyHostToDevice,
                c10::cuda::getCurrentCUDAStream()
            );
        }
    } else {
        op = op.device(torch::kCPU);
        tensor = torch::zeros({ channel, size.height, size.width }, op);
        for (auto i = 0; i < split_src.size(); ++i) {
            std::memcpy(
                tensor.data_ptr() + (size.width * size.height * sizeof(unsigned char) * i),
                split_src[i].data,
                sizeof(unsigned char) * size.width * size.height
            );
        }
    }
}

void normalize(torch::Tensor& tensor) {
    tensor /= 255.;
    tensor[0].sub_(0.485).div_(0.229);
    tensor[1].sub_(0.456).div_(0.224);
    tensor[2].sub_(0.406).div_(0.225);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Error!!" << std::endl;
        std::cerr << "<Usage>" << std::endl;
        std::cerr << "    ./example <modelfile> <src image file>" << std::endl;
        return 1;
    }
    // 画像をロード
    cv::Mat src = cv::imread(argv[2]);
    if (src.empty()) {
        std::cerr << "画像がロードできませんでした" << std::endl;
        return 1;
    }
    // スクリプトロード
    torch::jit::script::Module module = torch::jit::load(argv[1]);
    
    // 前処理
    cv::Mat crop_img;
    resize_and_crop(src, crop_img);
    // torch::Tensorに変換
    torch::Tensor input_tensor;
    mat_to_tensor(crop_img, input_tensor, true);
    input_tensor = input_tensor.to(torch::kFloat32); // float32にキャスト
    normalize(input_tensor); // 正規化
    input_tensor = input_tensor.unsqueeze(0); // バッチサイズ1にする

    // 推論
    auto scores = module.forward({ input_tensor });

    // 識別結果取得
    auto indices = scores.toTensor().argmax(1);
    indices = indices.to(torch::kCPU);
    auto index = indices[0].item().toInt();
    auto score = scores.toTensor().index({0, index}).item().toFloat();
    std::cout << index << ", " << score << std::endl;
    return 0;
}