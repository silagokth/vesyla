# Compilation ##################################################################
FROM archlinux:base as builder
WORKDIR /src
COPY . .
WORKDIR /src/build
RUN pacman -Syu --noconfirm python python-pip cmake gcc make boost protobuf && \
    pip3 install --break-system-packages --upgrade pip setuptools && \
    pip3 install --break-system-packages ortools protobuf==4.24.2 pyinstaller verboselogs coloredlogs numpy matplotlib binarytree sympy regex lark uuid svglib svg reportlab && \
    cmake -DCMAKE_INSTALL_PREFIX=/fakeroot .. && \
    make -j$(nproc) && \
    make install

# Test Image ###################################################################
FROM archlinux:base as tester
RUN pacman -Syu --noconfirm python python-pip gcc boost-libs && \
    pip3 install --break-system-packages --upgrade pip setuptools && \
    pip3 install --break-system-packages robotframework
COPY --from=builder /fakeroot /usr
WORKDIR /work
RUN vs-testcase generate /usr/share/vesyla/testcase && \
    sh run.sh

# Final Image ##################################################################
FROM archlinux:base
RUN pacman -Syu --noconfirm gcc boost-libs
COPY --from=builder /fakeroot /usr
WORKDIR /work
ENTRYPOINT [ "/bin/bash" ]
