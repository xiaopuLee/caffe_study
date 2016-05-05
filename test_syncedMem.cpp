#include <iostream>
#include "caffe/caffe.hpp"
#include "caffe/syncedmem.hpp"

#include "caffe/util/math_functions.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"

int main(int argc, char** argv)
{
    FLAGS_log_dir = "/home/sharp/caffe/tools/log/";
    google::InitGoogleLogging(argv[0]);

    google::ParseCommandLineFlags(&argc, &argv, true);

    caffe::SyncedMemory mem(10 * sizeof(int));
    CHECK_EQ(mem.head(), caffe::SyncedMemory::UNINITIALIZED);
    CHECK(mem.cpu_data());
    CHECK_EQ(mem.head(), caffe::SyncedMemory::HEAD_AT_CPU);
    CHECK(mem.gpu_data());
    CHECK_EQ(mem.head(), caffe::SyncedMemory::SYNCED);

    caffe::caffe_memset(mem.size(), 10, mem.mutable_cpu_data());


    LOG(INFO) << mem.size();


    return 0; 
}
