#!/bin/bash

ARCHITECTURE=$1
LIB_DIR=$2

VERSION="0.0.8"

PACKAGE_DIR="dist/diwa_${VERSION}_${ARCHITECTURE}"
DEBIAN_DIR="${PACKAGE_DIR}/DEBIAN"

USR_DIR="${PACKAGE_DIR}/usr"
INCLUDE_DIR="${USR_DIR}/src"
BUILD_DIR="dist/build"
SO_FILE="${BUILD_DIR}/libdiwa.so"

case "$ARCHITECTURE" in
    amd64)
        CROSS_COMPILE="g++ -fPIC"
        ;;
    riscv64)
        CROSS_COMPILE="riscv64-linux-gnu-gcc"
        ;;
    armhf)
        export PATH="/usr/lib/gcc/arm-none-eabi/13.2.1:$PATH"
        CROSS_COMPILE="arm-linux-gnueabihf-cpp"
        ;;
    *)
        echo -e "\033[93m[-]\033[0m Unsupported architecture: $ARCHITECTURE"
        exit 1
        ;;
esac

mkdir -p "${DEBIAN_DIR}"
mkdir -p "${INCLUDE_DIR}/diwa"
mkdir -p "${USR_DIR}/lib/${LIB_DIR}"
mkdir -p "${BUILD_DIR}"

echo -e "\033[92m[+]\033[0m Building shared library for ${ARCHITECTURE}..."
${CROSS_COMPILE} -shared -o "${SO_FILE}" -Isrc src/*.cpp

cp -r src/diwa.h "${INCLUDE_DIR}/"
cp "${SO_FILE}" "${USR_DIR}/lib/${LIB_DIR}/"

cat <<EOF > "${DEBIAN_DIR}/control"
Package: diwa
Version: ${VERSION}
Section: libs
Priority: optional
Architecture: ${ARCHITECTURE}
Depends: libc6 (>= 2.7)
Maintainer: Nathanne Isip <nathanneisip@gmail.com>
Description: diwa - Complex Compute Core Engine Library
 diwa (Complex Compute Core Engine) is a framework, platform,
 library, and an engine for handling complex computational
 tasks involving matrices, vectors, and tensors.
EOF

chmod 755 "${DEBIAN_DIR}"
chmod 755 "${USR_DIR}"
chmod 755 "${INCLUDE_DIR}"
chmod 755 "${USR_DIR}/lib/${LIB_DIR}"

dpkg-deb --build "${PACKAGE_DIR}" > /dev/null

rm -rf "${PACKAGE_DIR}"
rm -rf "${BUILD_DIR}"

echo -e "\033[92m[+]\033[0m Debian package for ${ARCHITECTURE} created successfully!"