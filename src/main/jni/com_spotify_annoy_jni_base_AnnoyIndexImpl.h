/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_spotify_annoy_jni_base_AnnoyIndexImpl */

#ifndef _Included_com_spotify_annoy_jni_base_AnnoyIndexImpl
#define _Included_com_spotify_annoy_jni_base_AnnoyIndexImpl
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppCtor
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppCtor
  (JNIEnv *, jobject, jint, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppDtor
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppDtor
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppAddItem
 * Signature: (JI[F)V
 */
JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppAddItem
  (JNIEnv *, jobject, jlong, jint, jfloatArray);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppGetNearestByVector
 * Signature: (J[FI)[I
 */
JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByVector
  (JNIEnv *, jobject, jlong, jfloatArray, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppGetNearestByVectorK
 * Signature: (J[FII)[I
 */
JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByVectorK
  (JNIEnv *, jobject, jlong, jfloatArray, jint, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppGetNearestByItem
 * Signature: (JII)[I
 */
JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByItem
  (JNIEnv *, jobject, jlong, jint, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppGetNearestByItemK
 * Signature: (JIII)[I
 */
JNIEXPORT jintArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetNearestByItemK
  (JNIEnv *, jobject, jlong, jint, jint, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppBuild
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppBuild
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppSave
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppSave
  (JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppLoad
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppLoad
  (JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppGetItemVector
 * Signature: (JI)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetItemVector
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppGetDistance
 * Signature: (JII)F
 */
JNIEXPORT jfloat JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppGetDistance
  (JNIEnv *, jobject, jlong, jint, jint);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppSize
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_spotify_annoy_jni_base_AnnoyIndexImpl
 * Method:    cppSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_spotify_annoy_jni_base_AnnoyIndexImpl_cppSetSeed
  (JNIEnv *, jobject, jlong, jint);

#ifdef __cplusplus
}
#endif
#endif
