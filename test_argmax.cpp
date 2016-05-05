#include <iostream>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/argmax_layer.hpp"
#include "caffe/common.hpp"

#include "caffe/proto/caffe.pb.h"
#include <boost/lexical_cast.hpp>

#include "glog/logging.h"
#include <vector>
#include <utility>

using namespace caffe;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "starting";
    caffe::Blob<float>* blob(new caffe::Blob<float>(2, 3, 1, 1));
    FillerParameter filler_pram;
    GaussianFiller<float> filler(filler_pram);

    filler.Fill(blob);

    caffe::Blob<float>* top_blob(new Blob<float>(2,1, 2, 1));
    std::vector<Blob<float>*> bottom;
    std::vector<Blob<float>*> top;
    bottom.push_back(blob);
    top.push_back(top_blob);
    LOG(INFO) << "start layer init";

    ArgMaxParameter argmax_param;
    LayerParameter layer_param;
    ArgMaxLayer<float>* argmax_layer(new ArgMaxLayer<float>(layer_param));
    argmax_layer->SetUp(bottom, top);
    LOG(INFO) << "Checking top shape";
    
    LOG(INFO) << "output bottom data";
    for(int i=0; i!=blob->shape(0); ++i)
    {
        std::string str = "";
        for(int j=0;j!=blob->shape(1); ++j)
        {
           str += boost::lexical_cast<std::string>(blob->data_at(i, j, 0, 0)) + " ";
        }
        LOG(INFO) << str;
    }
    CHECK_EQ(bottom[0]->shape().size(), top[0]->shape().size());
    float error = argmax_layer->Forward(bottom, top);
    LOG(INFO) << "the forward output is : " << error;

    for(int i=0; i!=top[0]->shape(0); ++i)
    {
        for(int j=0; j!=top[0]->shape(1); ++j)
        {
            LOG(INFO) << top[0]->data_at(i, j, 0, 0);
        }
    }
    LOG(INFO) << "finished!";
    return 0;
    

}
