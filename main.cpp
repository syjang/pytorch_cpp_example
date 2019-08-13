#include <torch/script.h>

#include <iostream>
#include <memory>
#include <vector>

int main(int agrc, char *argv[])
{
    auto module = torch::jit::load(argv[1]);
    std::vector<torch::jit::IValue> inputs;    
    auto t = torch::randn({1,3,512,512});    
    
    t = t.cuda();
    inputs.push_back(t);    
    module.to(torch::Device("cuda"));
    module.eval();
    
    c10::IValue output = module.forward(inputs);

    auto test = output.toTuple()->elements();
    std::cout << test.size() <<std::endl;
    for(int i =0 ;i < test.size() ; ++i)
    {
        torch::Tensor a = test[i].toTensor();
        std::cout << a.dim() << std::endl;
        std::cout << a[0][0] << std::endl;
    }
    return 0;
}