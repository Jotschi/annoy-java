#! /bin/bash
# Caution: chaos ahead

# Usage:
# ./build.sh annoy_hash 

# TODO: 
# * Verify args
# * make script more robust

pushd `dirname $0` > /dev/null
PROJECT_ROOT=`pwd`/..
popd > /dev/null

unamestr=`uname`
if [[ "$unamestr" == 'Darwin' ]]; then
   platform='mac'
elif [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
else
  echo "Platform $unamestr not supported"
  exit 1
fi

# FIXME: TMP_DIR=`mktemp -d` does not work with dockcross mount
TMP_DIR=$PROJECT_ROOT/.tmp_build_dir
mkdir $TMP_DIR
cd $TMP_DIR
chmod -R 777 ./

echo "[INFO] Pulling annoy sources..."
curl -sSO https://raw.githubusercontent.com/spotify/annoy/$1/src/annoylib.h
curl -sSO https://raw.githubusercontent.com/spotify/annoy/$1/src/kissrandom.h

if [ -z "$JAVA_HOME" ]; then
    if [[ "$platform" == 'mac' ]]; then
        JAVA_HOME=`/usr/libexec/java_home`
    elif [[ "$platform" == 'linux' ]]; then
        JAVA_HOME=/usr/lib/jvm/default-java
    fi
fi

cp -r $JAVA_HOME/include ./include

if [[ "$platform" == 'mac' ]]; then
    cp -r $JAVA_HOME/include/darwin/* ./include/
elif [[ "$platform" == 'linux' ]]; then
    cp -r $JAVA_HOME/include/linux/* ./include/
fi

cp -r $PROJECT_ROOT/native/* ./
cp -r $PROJECT_ROOT/src/main/java/* ./

echo "[INFO] Building jni headers..."
$JAVA_HOME/bin/javah -cp . -o ./com_spotify_annoy_jni_base_AnnoyIndexImpl.h -jni com.spotify.annoy.jni.base.AnnoyIndexImpl

if [[ "$platform" == 'mac' ]]; then
    echo "[INFO] Compiling Annoy code for mac-x64..."
    make > /dev/null
    mkdir $PROJECT_ROOT/target/classes/mac-x64
    mv libannoy.dylib $PROJECT_ROOT/target/classes/mac-x64/
fi

echo "[INFO] Compiling Annoy code for linux-x64 (make sure docker host is running)..."
./dockcross-linux-x64 bash -c "make > /dev/null"
mkdir $PROJECT_ROOT/target/classes/linux-x64
mv libannoy.so $PROJECT_ROOT/target/classes/linux-x64

# Cleanup
rm -fr $TMP_DIR
