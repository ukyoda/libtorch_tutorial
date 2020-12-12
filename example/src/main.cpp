#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <ATen/cuda/CUDAContext.h>

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
    // 入力データ
    auto op = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor input_tensor = torch::zeros({1, 3, 32, 32}, op);

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