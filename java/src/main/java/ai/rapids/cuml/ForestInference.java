package ai.rapids.cuml;

class ForestInference {
  // pointer to native ForestInference
  private final long handle;

  public ForestInference(int deviceId) {
    handle = nativeInit(deviceId);
  }

  public void transform(String cudaArrayInterface) {
    nativePredict(handle, cudaArrayInterface);
  }

  public void load(String modelPath) {
    nativeLoad(handle, modelPath);
  }


  private static native long nativeInit(int deviceId);

  private static native void nativePredict(long ptr, String cudaArrayInterface);

  private static native void nativeLoad(long ptr, String modelPath);
}
