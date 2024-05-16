# Compilation ##################################################################
FROM archlinux:base as builder
WORKDIR /src
COPY . .
WORKDIR /src/build
RUN pacman -Syu --noconfirm python python-pip cmake gcc make boost && \
    pip3 install --break-system-packages --upgrade pip setuptools && \
    pip3 install --break-system-packages ortools protobuf pyinstaller verboselogs coloredlogs numpy matplotlib binarytree sympy regex pyparsing && \
    cmake -DCMAKE_INSTALL_PREFIX=/fakeroot .. && \
    make -j$(nproc) && \
    make install

# Final Image ##################################################################
FROM archlinux:base
RUN pacman -Syu --noconfirm gcc boost-libs
COPY --from=builder /fakeroot /usr
WORKDIR /work
ENTRYPOINT [ "/bin/bash" ]
