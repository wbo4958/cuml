#include "cuml_fil.h"


class ForestInference {
public:
  ForestInference(int gpu_id): dev_id_(device_id) {
    handle = new raft::handle_t();
  }

  void predict();
  void load();

private:
  int dev_id_; 
  raft::handle_t *handle;
}


jlong JNICALL Java_ai_rapids_cuml_ForestInference_nativeInit(JNIEnv *env, jclass clazz, jint gpu_id) {
  ForestInference *fi = new ForestInference(gpu_id);
  return reinterpret_cast<jlong>(fi);
}

JNIEXPORT void JNICALL Java_ai_rapids_cuml_ForestInference_nativePredict
   (JNIEnv *env, jclass clazz, jlong ptr, jstring cuda_array_interface) {

  ForestInference *fi = reinterpret_cast<ForestInference *>(ptr);

  fi->predict();
  return;
}

