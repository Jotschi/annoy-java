#include <iostream>
#include <com_spotify_annoy_jni_base_AnnoyIndexImpl.h>
#include <jni.h>
#include <annoylib.h>
#include <kissrandom.h>

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  typedef AnnoyIndexMultiThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#else
  typedef AnnoyIndexSingleThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#endif

namespace
{

    inline jintArray vec_to_jintArray(JNIEnv *env, const vector<jint> &vec)
    {
        jintArray outJNIArray = env->NewIntArray(vec.size()); // allocate
        if (NULL == outJNIArray)
            return NULL;
        env->SetIntArrayRegion(outJNIArray, 0, vec.size(), &vec[0]); // copy
        return outJNIArray;
    }

    class HammingWrapper : public AnnoyIndexInterface<int32_t, float> {
        // Wrapper class for Hamming distance, using composition.
        // This translates binary (float) vectors into packed uint64_t vectors.
        // This is questionable from a performance point of view. Should reconsider this solution.
        private:
        int32_t _f_external, _f_internal;
        AnnoyIndex<int32_t, uint64_t, Hamming, Kiss64Random, AnnoyIndexThreadedBuildPolicy> _index;
        void _pack(const float* src, uint64_t* dst) const {
            for (int32_t i = 0; i < _f_internal; i++) {
            dst[i] = 0;
            for (int32_t j = 0; j < 64 && i*64+j < _f_external; j++) {
            dst[i] |= (uint64_t)(src[i * 64 + j] > 0.5) << j;
            }
            }
        };
        void _unpack(const uint64_t* src, float* dst) const {
            for (int32_t i = 0; i < _f_external; i++) {
            dst[i] = (src[i / 64] >> (i % 64)) & 1;
            }
        };
        public:
        HammingWrapper(int f) : _f_external(f), _f_internal((f + 63) / 64), _index((f + 63) / 64) {};
        bool add_item(int32_t item, const float* w, char**error) {
            vector<uint64_t> w_internal(_f_internal, 0);
            _pack(w, &w_internal[0]);
            return _index.add_item(item, &w_internal[0], error);
        };
        bool build(int q, int n_threads, char** error) { return _index.build(q, n_threads, error); };
        bool unbuild(char** error) { return _index.unbuild(error); };
        bool save(const char* filename, bool prefault, char** error) { return _index.save(filename, prefault, error); };
        void unload() { _index.unload(); };
        bool load(const char* filename, bool prefault, char** error) { return _index.load(filename, prefault, error); };
        float get_distance(int32_t i, int32_t j) const { return _index.get_distance(i, j); };
        void get_nns_by_item(int32_t item, size_t n, int search_k, vector<int32_t>* result, vector<float>* distances) const {
            if (distances) {
            vector<uint64_t> distances_internal;
            _index.get_nns_by_item(item, n, search_k, result, &distances_internal);
            distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
            } else {
            _index.get_nns_by_item(item, n, search_k, result, NULL);
            }
        };
        void get_nns_by_vector(const float* w, size_t n, int search_k, vector<int32_t>* result, vector<float>* distances) const {
            vector<uint64_t> w_internal(_f_internal, 0);
            _pack(w, &w_internal[0]);
            if (distances) {
            vector<uint64_t> distances_internal;
            _index.get_nns_by_vector(&w_internal[0], n, search_k, result, &distances_internal);
            distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
            } else {
            _index.get_nns_by_vector(&w_internal[0], n, search_k, result, NULL);
            }
        };
        int32_t get_n_items() const { return _index.get_n_items(); };
        int32_t get_n_trees() const { return _index.get_n_trees(); };
        void verbose(bool v) { _index.verbose(v); };
        void get_item(int32_t item, float* v) const {
            vector<uint64_t> v_internal(_f_internal, 0);
            _index.get_item(item, &v_internal[0]);
            _unpack(&v_internal[0], v);
        };
        void set_seed(int q) { _index.set_seed(q); };
        bool on_disk_build(const char* filename, char** error) { return _index.on_disk_build(filename, error); };
        };

    class ANN
    {
    public:
        jint f;
        AnnoyIndexInterface<jint, jfloat> *annoy_index;
        ANN(int f_, char metric) : f(f_)
        {
            switch (metric)
            {
            case 'h':
                annoy_index = new HammingWrapper(f);
                break;
            case 'a':
                annoy_index = new AnnoyIndex<jint, jfloat, Angular, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(f);
                break;
            case 'e':
                annoy_index = new AnnoyIndex<jint, jfloat, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(f);
                break;
            }
        }
        ~ANN()
        {
            delete annoy_index;
        }
    };
} // namespace


JNIEXPORT jlong JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppCtor(JNIEnv *env, jobject obj, jint jni_int, jint metric)
{
    return (intptr_t) new ANN(jni_int, metric);
}

JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppSetSeed(JNIEnv *env, jobject obj, jlong cpp_ptr, jint seed)
{
    ANN *ann = (ANN *)cpp_ptr;
    ann->annoy_index->set_seed(seed);
}

JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppAddItem(JNIEnv *env, jobject obj, jlong cpp_ptr, jint item, jfloatArray jni_floats)
{
    jfloat *inCArray = env->GetFloatArrayElements(jni_floats, NULL);
    if (NULL == inCArray)
        return;
    ANN *ann = (ANN *)cpp_ptr;
    ann->annoy_index->add_item(item, inCArray);
}

JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByVector(JNIEnv *env, jobject obj, jlong cpp_ptr, jfloatArray arr, jint n)
{
    jfloat *inCArray = env->GetFloatArrayElements(arr, NULL);
    if (NULL == inCArray)
        return NULL;
    size_t search_k = (size_t)-1;
    vector<jint> result;
    ANN *ann = (ANN *)cpp_ptr;
    ann->annoy_index->get_nns_by_vector(inCArray, n, search_k, &result, NULL);
    return vec_to_jintArray(env, result);
}

JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByVectorK(JNIEnv *env, jobject obj, jlong cpp_ptr, jfloatArray arr, jint n, jint search_k)
{
    jfloat *inCArray = env->GetFloatArrayElements(arr, NULL);
    if (NULL == inCArray)
        return NULL;
    vector<jint> result;
    ANN *ann = (ANN *)cpp_ptr;
    ann->annoy_index->get_nns_by_vector(inCArray, n, search_k, &result, NULL);
    return vec_to_jintArray(env, result);
}

JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByItem(JNIEnv *env, jobject obj, jlong cpp_ptr, jint item, jint n)
{
    size_t search_k = (size_t)-1;
    vector<jint> result;
    ANN *ann = (ANN *)cpp_ptr;
    ann->annoy_index->get_nns_by_item(item, n, search_k, &result, NULL);
    return vec_to_jintArray(env, result);
}

JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByItemK(JNIEnv *env, jobject obj, jlong cpp_ptr, jint item, jint n, jint search_k)
{
    vector<jint> result;
    ANN *ann = (ANN *)cpp_ptr;
    ann->annoy_index->get_nns_by_item(item, n, search_k, &result, NULL);
    return vec_to_jintArray(env, result);
}

JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppBuild(JNIEnv *env, jobject obj, jlong cpp_ptr, jint jni_int)
{
    ANN *ann = (ANN *)cpp_ptr;
    ann->annoy_index->build(jni_int);
}

JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppSave(JNIEnv *env, jobject obj, jlong cpp_ptr, jstring jni_filename)
{
    const char *filename = env->GetStringUTFChars(jni_filename, NULL);
    if (NULL == filename)
        return;
    ANN *ann = (ANN *)cpp_ptr;
    bool b = ann->annoy_index->save(filename);
    env->ReleaseStringUTFChars(jni_filename, filename); // release resources
}

JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppLoad(JNIEnv *env, jobject obj, jlong cpp_ptr, jstring jni_filename)
{
    const char *filename = env->GetStringUTFChars(jni_filename, NULL);
    if (NULL == filename)
        return;
    ANN *ann = (ANN *)cpp_ptr;
    bool b = ann->annoy_index->load(filename);
    env->ReleaseStringUTFChars(jni_filename, filename); // release resources
}

JNIEXPORT jfloatArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetItemVector(JNIEnv *env, jobject obj, jlong cpp_ptr, jint i)
{
    ANN *ann = (ANN *)cpp_ptr;
    vector<jfloat> vec(ann->f);
    ann->annoy_index->get_item(i, &vec[0]);
    jfloatArray outJNIArray = env->NewFloatArray(vec.size());
    if (NULL == outJNIArray)
        return NULL;
    env->SetFloatArrayRegion(outJNIArray, 0, vec.size(), &vec[0]); // copy
    return outJNIArray;
}

JNIEXPORT jfloat JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetDistance(JNIEnv *env, jobject obj, jlong cpp_ptr, jint jni_i, jint jni_j)
{
    ANN *ann = (ANN *)cpp_ptr;
    return (jfloat)ann->annoy_index->get_distance(jni_i, jni_j);
}

JNIEXPORT jint JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppSize(JNIEnv *env, jobject obj, jlong cpp_ptr)
{
    ANN *ann = (ANN *)cpp_ptr;
    return (jint)ann->annoy_index->get_n_items();
}

JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppDtor(JNIEnv *env, jobject obj, jlong cpp_ptr)
{
    ANN *ann = (ANN *)cpp_ptr;
    delete ann;
}
