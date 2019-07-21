/*
 * -\-\-
 * annoy-java
 * --
 * Copyright (C) 2016 Spotify AB
 * --
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * -/-/-
 */

package de.jotschi.annoy;

import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

@Properties({
	@Platform(

		includepath = { "src/main/jni", "annoy" },

		include = {

			"/home/jotschi/workspaces/mesh/annoy/src/annoylib.h"

		},

		link = "annoy"

	)
})
class AnnoyIndexImpl implements AnnoyIndex {

	private final int dim;
	// Stores the memory location of the tree in cpp that will be passed in
	// This is how we share states between java and cpp
	private final long cppPtr;

	public List<Integer> getNearestByVector(List<Float> vector, int nbNeighbors) {
		validateVecSize(vector);
		return primitiveToBoxed(
			cppGetNearestByVector(cppPtr, boxedToPrimitive(vector), nbNeighbors));
	}

	public List<Integer> getNearestByVector(List<Float> vector, int nbNeighbors, int searchK) {
		validateVecSize(vector);
		return primitiveToBoxed(
			cppGetNearestByVectorK(cppPtr, boxedToPrimitive(vector), nbNeighbors, searchK));
	}

	public List<Integer> getNearestByItem(int item, int nbNeighbors) {
		return primitiveToBoxed(cppGetNearestByItem(cppPtr, item, nbNeighbors));
	}

	public List<Integer> getNearestByItem(int item, int nbNeighbors, int searchK) {
		return primitiveToBoxed(cppGetNearestByItemK(cppPtr, item, nbNeighbors, searchK));
	}

	public AnnoyIndex save(String filename) {
		cppSave(cppPtr, filename);
		return this;
	}

	public void close() {
		cppDtor(cppPtr);
	}

	public List<Float> getItemVector(int item) {
		return primitiveToBoxed(cppGetItemVector(cppPtr, item));
	}

	public float getDistance(int itemA, int itemB) {
		return cppGetDistance(cppPtr, itemA, itemB);
	}

	public int size() {
		return cppSize(cppPtr);
	}

	// Construction

	AnnoyIndexImpl(int dim, Annoy.Metric angular) {
		this.dim = dim;
		System.load(Annoy.ANNOY_LIB_PATH);
		this.cppPtr = cppCtor(dim, angular.name().toLowerCase().charAt(0));
	}

	AnnoyIndexImpl addItem(int item, List<Float> vector) {
		validateVecSize(vector);
		cppAddItem(cppPtr, item, boxedToPrimitive(vector));
		return this;
	}

	AnnoyIndexImpl addAllItems(Iterable<List<Float>> vectors) {
		int nb = size();
		for (List<Float> vector : vectors) {
			addItem(nb++, vector);
		}
		return this;
	}

	AnnoyIndexImpl build(int nbTrees) {
		cppBuild(cppPtr, nbTrees);
		return this;
	}

	AnnoyIndexImpl load(String filename) throws FileNotFoundException {
		if (Files.notExists(Paths.get(filename))) {
			throw new FileNotFoundException("Cannot find annoy index: " + filename);
		}
		cppLoad(cppPtr, filename);
		return this;
	}

	AnnoyIndexImpl setSeed(int seed) {
		cppSetSeed(cppPtr, seed);
		return this;
	}

	// Helpers

	private static List<Float> primitiveToBoxed(float[] vector) {
		return Arrays.asList(ArrayUtils.toObject(vector));
	}

	private static List<Integer> primitiveToBoxed(int[] vector) {
		return Arrays.asList(ArrayUtils.toObject(vector));
	}

	private static float[] boxedToPrimitive(List<Float> vector) {
		return ArrayUtils.toPrimitive(vector.toArray(new Float[0]));
	}

	private void validateVecSize(List<Float> vector) {
		if (vector.size() != dim) {
			throw new RuntimeException("Item's vector should match the dimension of the tree");
		}
	}

	// Native cpp methods

	/**
	private native void cppAddItem(long cppPtr, int item, float[] vector);
	  virtual bool add_item(S item, const T* w, char** error=NULL) = 0;

	private native void cppBuild(long cppPtr, int nbTrees);
	  virtual bool build(int q, char** error=NULL) = 0;
	
	  virtual bool unbuild(char** error=NULL) = 0;

	private native void cppSave(long cppPtr, String filename);
	  virtual bool save(const char* filename, bool prefault=false, char** error=NULL) = 0;
	
	  virtual void unload() = 0;

	private native void cppLoad(long cppPtr, String filename);
	  virtual bool load(const char* filename, bool prefault=false, char** error=NULL) = 0;
	
	private native float cppGetDistance(long cppPtr, int itemA, int itemB);
	  virtual T get_distance(S i, S j) const = 0;

	  virtual void get_nns_by_item(S item, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const = 0;

	private native int[] cppGetNearestByVector(long cppPtr, float[] vector, int nbNeighbors);
	  virtual void get_nns_by_vector(const T* w, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const = 0;
	
	  virtual S get_n_items() const = 0;
	  virtual S get_n_trees() const = 0;
	  virtual void verbose(bool v) = 0;

	private native float[] cppGetItemVector(long cppPtr, int item);
	  virtual void get_item(S item, T* v) const = 0;

	private native void cppSetSeed(long cppPtr, int seed);
	  virtual void set_seed(int q) = 0;
	
	  virtual bool on_disk_build(const char* filename, char** error=NULL) = 0;

**/  
	
	
	private native int[] cppGetNearestByVectorK(long cppPtr, float[] vector, int nbNeighbors,		int searchK);
	private native int cppSize(long cppPtr);
	// returns the c++ memory index for the object
	private native long cppCtor(int dim, int metric);

	private native void cppDtor(long cppPtr);

	private native int[] cppGetNearestByItem(long cppPtr, int item, int nbNeighbors);

	private native int[] cppGetNearestByItemK(long cppPtr, int item, int nbNeighbors, int searchK);



}
